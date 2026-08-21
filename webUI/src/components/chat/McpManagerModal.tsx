import React, { useState } from 'react';
import { Modal } from '../ui/Modal';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { Input } from '../ui/Input';
import { Switch } from '../ui/Switch';
import { mcpService } from '../../services/mcp';
import { McpServer } from '../../types';

interface McpManagerModalProps {
  isOpen: boolean;
  onClose: () => void;
  onToolsUpdated: () => void;
}

export const McpManagerModal: React.FC<McpManagerModalProps> = ({
  isOpen,
  onClose,
  onToolsUpdated,
}) => {
  const [servers, setServers] = useState<McpServer[]>(() => mcpService.getServers());
  const [showAddServer, setShowAddServer] = useState(false);
  const [newServerName, setNewServerName] = useState('');
  const [newServerCommand, setNewServerCommand] = useState('');
  const [newServerType, setNewServerType] = useState<'stdio' | 'sse'>('stdio');

  const refreshServers = () => {
    setServers(mcpService.getServers());
    onToolsUpdated();
  };

  const handleToggleTool = (serverId: string, toolName: string, current: boolean) => {
    mcpService.toggleTool(serverId, toolName, !current);
    refreshServers();
  };

  const handleAddServer = () => {
    if (!newServerName.trim()) return;
    mcpService.addServer({
      name: newServerName.trim(),
      type: newServerType,
      command: newServerCommand.trim(),
    });
    setNewServerName('');
    setNewServerCommand('');
    setShowAddServer(false);
    refreshServers();
  };

  const handleRemoveServer = (id: string) => {
    mcpService.removeServer(id);
    refreshServers();
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Model Context Protocol (MCP) Manager"
      maxWidth="2xl"
    >
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <div className="space-y-0.5">
            <p className="text-xs text-muted-foreground">
              Connect external tools, file systems, databases, and APIs via the open Model Context Protocol standard.
            </p>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowAddServer(!showAddServer)}
          >
            {showAddServer ? 'Cancel' : '+ Add MCP Server'}
          </Button>
        </div>

        {/* Add Server Drawer */}
        {showAddServer && (
          <div className="p-4 rounded-2xl border border-teal-500/30 bg-secondary/70 space-y-3 animate-in fade-in duration-150">
            <h4 className="text-xs font-bold text-foreground">Configure New MCP Server</h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <Input
                label="Server Name"
                value={newServerName}
                onChange={(e) => setNewServerName(e.target.value)}
                placeholder="e.g. GitHub MCP Server"
              />
              <div className="space-y-1.5">
                <label className="block text-xs font-medium text-foreground">Transport Protocol</label>
                <select
                  value={newServerType}
                  onChange={(e) => setNewServerType(e.target.value as 'stdio' | 'sse')}
                  className="w-full bg-background border border-border rounded-xl px-3 py-2 text-xs font-mono text-foreground focus:outline-none focus:border-teal-500/50"
                >
                  <option value="stdio">STDIO (Local Command Process)</option>
                  <option value="sse">SSE (HTTP Server-Sent Events)</option>
                </select>
              </div>
            </div>

            <Input
              label={newServerType === 'stdio' ? 'Command Line Execution' : 'SSE Endpoint URL'}
              value={newServerCommand}
              onChange={(e) => setNewServerCommand(e.target.value)}
              placeholder={newServerType === 'stdio' ? 'npx -y @modelcontextprotocol/server-github' : 'http://localhost:8000/sse'}
            />

            <div className="flex justify-end pt-1">
              <Button variant="primary" size="sm" onClick={handleAddServer}>
                Connect Server
              </Button>
            </div>
          </div>
        )}

        {/* Active Servers & Tools List */}
        <div className="space-y-4 max-h-[55vh] overflow-y-auto pr-1">
          {servers.map((server) => (
            <div
              key={server.id}
              className="p-4 rounded-2xl border border-border bg-card space-y-3 shadow-xs"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <div className="p-2 rounded-xl bg-teal-500/10 text-teal-500 border border-teal-500/20">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <rect width="18" height="18" x="3" y="3" rx="2" />
                      <path d="M7 7h10M7 12h10M7 17h10" />
                    </svg>
                  </div>
                  <div>
                    <h4 className="font-bold text-xs text-foreground">{server.name}</h4>
                    <span className="text-[10px] font-mono text-muted-foreground">{server.command || server.url || 'Native Provider'}</span>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <Badge variant="success" size="sm">
                    {server.status}
                  </Badge>
                  {server.id !== 'system-tools' && (
                    <button
                      onClick={() => handleRemoveServer(server.id)}
                      className="text-muted-foreground hover:text-red-500 p-1 text-xs"
                      title="Remove Server"
                    >
                      ×
                    </button>
                  )}
                </div>
              </div>

              {/* Tools list */}
              <div className="space-y-2 pt-2 border-t border-border">
                <span className="text-[10px] font-mono font-bold text-muted-foreground uppercase tracking-wider block">
                  AVAILABLE TOOLS ({server.tools.length})
                </span>
                <div className="space-y-1.5">
                  {server.tools.map((tool) => {
                    const isEnabled = tool.enabled !== false;
                    return (
                      <div
                        key={tool.name}
                        className="p-2.5 rounded-xl bg-secondary border border-border flex items-start justify-between gap-3 text-xs"
                      >
                        <div className="space-y-0.5 max-w-md">
                          <div className="flex items-center gap-1.5">
                            <span className="font-mono font-bold text-teal-600 dark:text-teal-400">{tool.name}</span>
                          </div>
                          <p className="text-[11px] text-muted-foreground leading-relaxed">{tool.description}</p>
                        </div>

                        <Switch
                          checked={isEnabled}
                          onChange={() => handleToggleTool(server.id, tool.name, isEnabled)}
                        />
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </Modal>
  );
};
