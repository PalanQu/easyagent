export interface StandaloneConfig {
  apiBaseUrl: string;
  copilotkitPath?: string;
  agentName?: string;
  apiKey?: string;
  userId?: string;
  userName?: string;
  userEmail?: string;
}

const CONFIG_KEY = "deep-agent-config";

export function getConfig(): StandaloneConfig | null {
  if (typeof window === "undefined") return null;

  const stored = localStorage.getItem(CONFIG_KEY);
  if (!stored) return null;

  try {
    const parsed = JSON.parse(stored) as Partial<
      StandaloneConfig & { deploymentUrl?: string; assistantId?: string; langsmithApiKey?: string }
    >;
    return {
      apiBaseUrl: parsed.apiBaseUrl || parsed.deploymentUrl || "",
      copilotkitPath: parsed.copilotkitPath || "/copilotkit",
      agentName: parsed.agentName || parsed.assistantId || "easyagent",
      apiKey: parsed.apiKey || parsed.langsmithApiKey,
      userId: parsed.userId,
      userName: parsed.userName,
      userEmail: parsed.userEmail,
    };
  } catch {
    return null;
  }
}

export function saveConfig(config: StandaloneConfig): void {
  if (typeof window === "undefined") return;
  localStorage.setItem(CONFIG_KEY, JSON.stringify(config));
}
