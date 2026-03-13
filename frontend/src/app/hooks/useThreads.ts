import useSWRInfinite from "swr/infinite";
import { getConfig } from "@/lib/config";

export type ThreadStatus = "idle" | "busy" | "interrupted" | "error";

export interface ThreadItem {
  id: string;
  updatedAt: Date;
  status: ThreadStatus;
  title: string;
  description: string;
}

type SessionOut = {
  id: number;
  thread_id: string | null;
  created_at: string;
  session_context: Record<string, unknown>;
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

function toThreadItem(session: SessionOut): ThreadItem {
  const context = session.session_context || {};
  const titleRaw = context.title;
  const descriptionRaw = context.description;
  const title = typeof titleRaw === "string" && titleRaw.trim() ? titleRaw : `Session ${session.id}`;
  const description =
    typeof descriptionRaw === "string" && descriptionRaw.trim()
      ? descriptionRaw
      : session.thread_id || "No description";
  return {
    id: String(session.id),
    updatedAt: new Date(session.created_at),
    status: "idle",
    title,
    description,
  };
}

const DEFAULT_PAGE_SIZE = 20;

export function useThreads(props: { status?: ThreadStatus; limit?: number }) {
  const pageSize = props.limit || DEFAULT_PAGE_SIZE;

  return useSWRInfinite(
    (pageIndex: number) => {
      const config = getConfig();
      if (!config?.apiBaseUrl) return null;
      if (pageIndex > 0) return null; // Backend endpoint currently returns full list.
      return {
        kind: "threads" as const,
        apiBaseUrl: config.apiBaseUrl.replace(/\/$/, ""),
        status: props.status,
        pageSize,
      };
    },
    async ({
      apiBaseUrl,
      status,
      pageSize,
    }: {
      kind: "threads";
      apiBaseUrl: string;
      status?: ThreadStatus;
      pageSize: number;
    }) => {
      const response = await fetch(`${apiBaseUrl}/me/sessions`, {
        headers: buildHeaders(),
      });
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}`);
      }

      const sessions = (await response.json()) as SessionOut[];
      let items = sessions.map(toThreadItem).sort((a, b) => b.updatedAt.getTime() - a.updatedAt.getTime());
      if (status) {
        items = items.filter((item) => item.status === status);
      }
      return items.slice(0, pageSize);
    },
    {
      revalidateFirstPage: true,
      revalidateOnFocus: true,
    }
  );
}
