"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useQueryState } from "nuqs";
import { v4 as uuidv4 } from "uuid";
import { getConfig } from "@/lib/config";
import type { ChatMessage } from "@/app/types/types";
import type { TodoItem } from "@/app/types/types";

export type StateType = {
  messages: ChatMessage[];
  todos: TodoItem[];
  files: Record<string, string>;
  email?: {
    id?: string;
    subject?: string;
    page_content?: string;
  };
  ui?: any;
};

type SessionOut = {
  id: number;
  thread_id: string | null;
};

type SessionMessageOut = {
  role: "user" | "assistant" | "tool" | string;
  content: string;
};

type SSEPacket = {
  event?: string;
  data: string;
};

function buildHeaders() {
  const config = getConfig();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (config?.apiKey) {
    headers["X-Api-Key"] = config.apiKey;
  }
  if (config?.userId) {
    headers["X-User-ID"] = config.userId;
  }
  if (config?.userName) {
    headers["X-User-Name"] = config.userName;
  }
  if (config?.userEmail) {
    headers["X-User-Email"] = config.userEmail;
  }
  return headers;
}

function mapRoleToType(role: string): ChatMessage["type"] {
  if (role === "assistant") return "ai";
  if (role === "tool") return "tool";
  return "human";
}

function mapSessionMessages(messages: SessionMessageOut[]): ChatMessage[] {
  return messages.map((message, index) => ({
    id: `session-msg-${index}-${uuidv4()}`,
    type: mapRoleToType(message.role),
    content: message.content,
  }));
}

async function parseError(response: Response): Promise<string> {
  try {
    const body = await response.json();
    if (typeof body?.detail === "string") {
      return body.detail;
    }
  } catch {
    // Ignore parse errors and fallback to status text.
  }
  return `${response.status} ${response.statusText}`;
}

function parseSSEPackets(chunk: string): SSEPacket[] {
  const packets: SSEPacket[] = [];
  const frames = chunk.split("\n\n");
  for (const frame of frames) {
    if (!frame.trim()) continue;
    let event: string | undefined;
    const dataLines: string[] = [];
    const lines = frame.split("\n");
    for (const line of lines) {
      if (line.startsWith("event:")) {
        event = line.slice("event:".length).trim();
      } else if (line.startsWith("data:")) {
        dataLines.push(line.slice("data:".length).trim());
      }
    }
    if (dataLines.length === 0) continue;
    packets.push({ event, data: dataLines.join("\n") });
  }
  return packets;
}

function extractTextBlocks(value: unknown): string[] {
  if (typeof value === "string") {
    return value.trim() ? [value] : [];
  }

  if (!value || typeof value !== "object") {
    return [];
  }

  if (Array.isArray(value)) {
    return value.flatMap((item) => extractTextBlocks(item));
  }

  const obj = value as Record<string, unknown>;
  if (typeof obj.role === "string" && !["assistant", "ai"].includes(obj.role)) {
    return [];
  }

  const blocks: string[] = [];
  const directKeys = ["delta", "text", "content"] as const;
  for (const key of directKeys) {
    const field = obj[key];
    if (typeof field === "string" && field.trim()) {
      blocks.push(field);
    } else if (Array.isArray(field)) {
      for (const item of field) {
        if (typeof item === "string" && item.trim()) {
          blocks.push(item);
        } else if (
          item &&
          typeof item === "object" &&
          (item as Record<string, unknown>).type === "text" &&
          typeof (item as Record<string, unknown>).text === "string"
        ) {
          blocks.push(String((item as Record<string, unknown>).text));
        }
      }
    }
  }

  if (obj.message) {
    blocks.push(...extractTextBlocks(obj.message));
  }
  if (obj.output) {
    blocks.push(...extractTextBlocks(obj.output));
  }
  if (obj.result) {
    blocks.push(...extractTextBlocks(obj.result));
  }

  return blocks;
}

export function useChat({ onHistoryRevalidate }: { onHistoryRevalidate?: () => void }) {
  const [threadId, setThreadId] = useQueryState("threadId");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isThreadLoading, setIsThreadLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const config = getConfig();
  const apiBaseUrl = config?.apiBaseUrl?.replace(/\/$/, "") || "";
  const copilotkitPath = config?.copilotkitPath || "/copilotkit";

  const loadThreadMessages = useCallback(
    async (sessionId: string) => {
      if (!apiBaseUrl) return;
      setIsThreadLoading(true);
      try {
        const response = await fetch(
          `${apiBaseUrl}/me/sessions/${sessionId}/messages`,
          { headers: buildHeaders() }
        );
        if (!response.ok) {
          throw new Error(await parseError(response));
        }
        const payload = (await response.json()) as SessionMessageOut[];
        setMessages(mapSessionMessages(payload));
      } finally {
        setIsThreadLoading(false);
      }
    },
    [apiBaseUrl]
  );

  useEffect(() => {
    if (!threadId) {
      setMessages([]);
      return;
    }
    void loadThreadMessages(threadId);
  }, [threadId, loadThreadMessages]);

  const createSession = useCallback(async (): Promise<SessionOut> => {
    const response = await fetch(`${apiBaseUrl}/me/sessions`, {
      method: "POST",
      headers: buildHeaders(),
      body: JSON.stringify({
        thread_id: uuidv4(),
        session_context: {},
      }),
    });
    if (!response.ok) {
      throw new Error(await parseError(response));
    }
    return (await response.json()) as SessionOut;
  }, [apiBaseUrl]);

  const sendMessage = useCallback(
    async (content: string) => {
      if (!apiBaseUrl || isLoading) return;
      setIsLoading(true);
      try {
        let sessionId = threadId;
        let threadIdentifier: string;

        if (!sessionId) {
          const created = await createSession();
          sessionId = String(created.id);
          threadIdentifier = created.thread_id || uuidv4();
          await setThreadId(sessionId);
          onHistoryRevalidate?.();
        } else {
          const meSessionRes = await fetch(`${apiBaseUrl}/me/sessions/${sessionId}`, {
            headers: buildHeaders(),
          });
          if (!meSessionRes.ok) {
            throw new Error(await parseError(meSessionRes));
          }
          const existing = (await meSessionRes.json()) as SessionOut;
          threadIdentifier = existing.thread_id || uuidv4();
        }

        const optimisticMessage: ChatMessage = {
          id: uuidv4(),
          type: "human",
          content,
        };
        setMessages((prev) => [...prev, optimisticMessage]);

        const controller = new AbortController();
        abortRef.current = controller;
        const assistantMessageId = uuidv4();
        let assistantCreated = false;
        let assistantBuffer = "";

        const copilotResponse = await fetch(`${apiBaseUrl}${copilotkitPath}`, {
          method: "POST",
          headers: {
            ...buildHeaders(),
            Accept: "text/event-stream",
          },
          signal: controller.signal,
          body: JSON.stringify({
            threadId: threadIdentifier,
            runId: uuidv4(),
            state: {},
            messages: [
              {
                id: uuidv4(),
                role: "user",
                content: [{ type: "text", text: content }],
              },
            ],
            tools: [],
            context: [],
            forwardedProps: {},
          }),
        });
        if (!copilotResponse.ok) {
          throw new Error(await parseError(copilotResponse));
        }

        // Drain stream to completion; UI refreshes from persisted thread state afterwards.
        if (copilotResponse.body) {
          const reader = copilotResponse.body.getReader();
          const decoder = new TextDecoder();
          let pending = "";

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            pending += decoder.decode(value, { stream: true });
            const lastBoundary = pending.lastIndexOf("\n\n");
            if (lastBoundary === -1) continue;

            const complete = pending.slice(0, lastBoundary);
            pending = pending.slice(lastBoundary + 2);
            const packets = parseSSEPackets(complete);

            for (const packet of packets) {
              if (packet.data === "[DONE]") continue;
              let parsed: unknown = packet.data;
              try {
                parsed = JSON.parse(packet.data);
              } catch {
                // Keep raw string fallback.
              }
              const parts = extractTextBlocks(parsed);
              if (parts.length === 0) continue;

              if (!assistantCreated) {
                assistantCreated = true;
                setMessages((prev) => [
                  ...prev,
                  { id: assistantMessageId, type: "ai", content: "" },
                ]);
              }

              assistantBuffer += parts.join("");
              const nextValue = assistantBuffer;
              setMessages((prev) =>
                prev.map((message) =>
                  message.id === assistantMessageId
                    ? { ...message, content: nextValue }
                    : message
                )
              );
            }
          }
        }

        if (sessionId) {
          await loadThreadMessages(sessionId);
          onHistoryRevalidate?.();
        }
      } catch (error) {
        const message = error instanceof Error ? error.message : "Request failed";
        setMessages((prev) => [
          ...prev,
          {
            id: uuidv4(),
            type: "ai",
            content: `Error: ${message}`,
          },
        ]);
      } finally {
        abortRef.current = null;
        setIsLoading(false);
      }
    },
    [
      apiBaseUrl,
      copilotkitPath,
      createSession,
      isLoading,
      loadThreadMessages,
      onHistoryRevalidate,
      setThreadId,
      threadId,
    ]
  );

  const stopStream = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsLoading(false);
  }, []);

  const setFiles = useCallback(async (_files: Record<string, string>) => {}, []);
  const resumeInterrupt = useCallback((_value: any) => {}, []);
  const interrupt: { value: any } | undefined = undefined;

  return {
    stream: undefined,
    todos: [] as TodoItem[],
    files: {} as Record<string, string>,
    email: undefined,
    ui: [] as any[],
    setFiles,
    messages,
    isLoading,
    isThreadLoading,
    interrupt,
    getMessagesMetadata: () => ({}),
    sendMessage,
    runSingleStep: () => {},
    continueStream: () => {},
    stopStream,
    markCurrentThreadAsResolved: () => {},
    resumeInterrupt,
  };
}
