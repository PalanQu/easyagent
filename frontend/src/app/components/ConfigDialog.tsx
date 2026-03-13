"use client";

import { useEffect, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { StandaloneConfig } from "@/lib/config";

interface ConfigDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onSave: (config: StandaloneConfig) => void;
  initialConfig?: StandaloneConfig;
}

export function ConfigDialog({
  open,
  onOpenChange,
  onSave,
  initialConfig,
}: ConfigDialogProps) {
  const [apiBaseUrl, setApiBaseUrl] = useState(initialConfig?.apiBaseUrl || "");
  const [copilotkitPath, setCopilotkitPath] = useState(initialConfig?.copilotkitPath || "/copilotkit");
  const [agentName, setAgentName] = useState(initialConfig?.agentName || "easyagent");
  const [apiKey, setApiKey] = useState(initialConfig?.apiKey || "");
  const [userId, setUserId] = useState(initialConfig?.userId || "");
  const [userName, setUserName] = useState(initialConfig?.userName || "");
  const [userEmail, setUserEmail] = useState(initialConfig?.userEmail || "");

  useEffect(() => {
    if (!open || !initialConfig) return;
    setApiBaseUrl(initialConfig.apiBaseUrl || "");
    setCopilotkitPath(initialConfig.copilotkitPath || "/copilotkit");
    setAgentName(initialConfig.agentName || "easyagent");
    setApiKey(initialConfig.apiKey || "");
    setUserId(initialConfig.userId || "");
    setUserName(initialConfig.userName || "");
    setUserEmail(initialConfig.userEmail || "");
  }, [open, initialConfig]);

  const handleSave = () => {
    if (!apiBaseUrl.trim()) {
      alert("Please fill in API Base URL");
      return;
    }

    onSave({
      apiBaseUrl: apiBaseUrl.trim(),
      copilotkitPath: copilotkitPath.trim() || "/copilotkit",
      agentName: agentName.trim() || "easyagent",
      apiKey: apiKey.trim() || undefined,
      userId: userId.trim() || undefined,
      userName: userName.trim() || undefined,
      userEmail: userEmail.trim() || undefined,
    });
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[560px]">
        <DialogHeader>
          <DialogTitle>Configuration</DialogTitle>
          <DialogDescription>
            Configure EasyAgent backend settings. These values are saved in local storage.
          </DialogDescription>
        </DialogHeader>
        <div className="grid gap-4 py-4">
          <div className="grid gap-2">
            <Label htmlFor="apiBaseUrl">API Base URL</Label>
            <Input
              id="apiBaseUrl"
              placeholder="http://127.0.0.1:8000"
              value={apiBaseUrl}
              onChange={(e) => setApiBaseUrl(e.target.value)}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="copilotkitPath">CopilotKit Path</Label>
            <Input
              id="copilotkitPath"
              placeholder="/copilotkit"
              value={copilotkitPath}
              onChange={(e) => setCopilotkitPath(e.target.value)}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="agentName">
              Agent Name <span className="text-muted-foreground">(Optional)</span>
            </Label>
            <Input
              id="agentName"
              placeholder="easyagent"
              value={agentName}
              onChange={(e) => setAgentName(e.target.value)}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="apiKey">
              API Key <span className="text-muted-foreground">(Optional)</span>
            </Label>
            <Input
              id="apiKey"
              type="password"
              placeholder="your-api-key"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="userId">
              X-User-ID <span className="text-muted-foreground">(Optional)</span>
            </Label>
            <Input
              id="userId"
              placeholder="alice"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="userName">
              X-User-Name <span className="text-muted-foreground">(Optional)</span>
            </Label>
            <Input
              id="userName"
              placeholder="Alice"
              value={userName}
              onChange={(e) => setUserName(e.target.value)}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="userEmail">
              X-User-Email <span className="text-muted-foreground">(Optional)</span>
            </Label>
            <Input
              id="userEmail"
              placeholder="alice@example.com"
              value={userEmail}
              onChange={(e) => setUserEmail(e.target.value)}
            />
          </div>
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={handleSave}>Save</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
