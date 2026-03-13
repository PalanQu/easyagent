"use client";

import { ReactNode, createContext, useContext } from "react";
import { useChat } from "@/app/hooks/useChat";

interface ChatProviderProps {
  children: ReactNode;
  onHistoryRevalidate?: () => void;
}

export function ChatProvider({ children, onHistoryRevalidate }: ChatProviderProps) {
  const chat = useChat({ onHistoryRevalidate });
  return <ChatContext.Provider value={chat}>{children}</ChatContext.Provider>;
}

export type ChatContextType = ReturnType<typeof useChat>;

export const ChatContext = createContext<ChatContextType | undefined>(
  undefined
);

export function useChatContext() {
  const context = useContext(ChatContext);
  if (context === undefined) {
    throw new Error("useChatContext must be used within a ChatProvider");
  }
  return context;
}
