export type ChatMessageType = "human" | "ai" | "tool";

export interface ChatMessage {
  id: string;
  type: ChatMessageType;
  content: string | Array<string | { type?: string; text?: string }>;
  tool_calls?: Array<{
    id?: string;
    name?: string;
    args?: Record<string, unknown>;
    function?: { name?: string; arguments?: unknown };
    input?: unknown;
    type?: string;
  }>;
  additional_kwargs?: {
    tool_calls?: Array<{
      id?: string;
      function?: { name?: string; arguments?: unknown };
      name?: string;
      args?: unknown;
      input?: unknown;
      type?: string;
    }>;
  };
  tool_call_id?: string;
  name?: string;
}

export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
  result?: string;
  status: "pending" | "completed" | "error" | "interrupted";
}

export interface SubAgent {
  id: string;
  name: string;
  subAgentName: string;
  input: Record<string, unknown>;
  output?: Record<string, unknown>;
  status: "pending" | "active" | "completed" | "error";
}

export interface FileItem {
  path: string;
  content: string;
}

export interface TodoItem {
  id: string;
  content: string;
  status: "pending" | "in_progress" | "completed";
  updatedAt?: Date;
}

export interface Thread {
  id: string;
  title: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface InterruptData {
  value: any;
  ns?: string[];
  scope?: string;
}

export interface ActionRequest {
  name: string;
  args: Record<string, unknown>;
  description?: string;
}

export interface ReviewConfig {
  actionName: string;
  allowedDecisions?: string[];
}

export interface ToolApprovalInterruptData {
  action_requests: ActionRequest[];
  review_configs?: ReviewConfig[];
}
