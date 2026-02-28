import { useState } from 'react'

interface SettingsModalProps {
  onClose: () => void
}

const LLM_PROVIDERS = [
  { id: 'openai', name: 'OpenAI', placeholder: 'sk-...', defaultBase: 'https://api.openai.com/v1', models: ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo'] },
  { id: 'anthropic', name: 'Anthropic', placeholder: 'sk-ant-...', defaultBase: 'https://api.anthropic.com', models: ['claude-sonnet-4-20250514', 'claude-haiku-4-20250414'] },
  { id: 'qwen', name: 'Qwen', placeholder: 'sk-...', defaultBase: 'https://dashscope.aliyuncs.com/compatible-mode/v1', models: ['qwen-max', 'qwen-plus', 'qwen-turbo'] },
  { id: 'kimi', name: 'Kimi', placeholder: 'sk-...', defaultBase: 'https://api.moonshot.cn/v1', models: ['moonshot-v1-128k', 'moonshot-v1-32k', 'moonshot-v1-8k'] },
  { id: 'minimax', name: 'MiniMax', placeholder: 'eyJ...', defaultBase: 'https://api.minimax.chat/v1', models: ['abab6.5s-chat', 'abab5.5-chat'] },
  { id: 'gemini', name: 'Gemini', placeholder: 'AI...', defaultBase: 'https://generativelanguage.googleapis.com/v1beta/openai/', models: ['gemini-2.0-flash', 'gemini-2.0-pro'] },
]

export default function SettingsModal({ onClose }: SettingsModalProps) {
  const [provider, setProvider] = useState('openai')
  const [apiKey, setApiKey] = useState('')
  const [baseUrl, setBaseUrl] = useState(LLM_PROVIDERS[0].defaultBase)
  const [model, setModel] = useState(LLM_PROVIDERS[0].models[0])

  const current = LLM_PROVIDERS.find(p => p.id === provider)!

  const handleProviderChange = (id: string) => {
    const p = LLM_PROVIDERS.find(x => x.id === id)!
    setProvider(id)
    setBaseUrl(p.defaultBase)
    setModel(p.models[0])
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/20" onClick={onClose} />
      <div className="relative bg-surface rounded-xl shadow-xl w-full max-w-lg mx-4 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border-light">
          <h2 className="text-sm font-semibold text-text-primary">Settings</h2>
          <button
            onClick={onClose}
            className="text-text-tertiary hover:text-text-primary p-1 rounded hover:bg-surface-tertiary transition-colors"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="px-5 py-4 space-y-4">
          {/* Provider selection */}
          <div>
            <label className="text-xs font-medium text-text-secondary mb-2 block">LLM Provider</label>
            <div className="grid grid-cols-3 gap-1.5">
              {LLM_PROVIDERS.map(p => (
                <button
                  key={p.id}
                  onClick={() => handleProviderChange(p.id)}
                  className={`text-xs py-2 px-3 rounded-lg border transition-colors ${
                    provider === p.id
                      ? 'border-accent bg-accent-light text-accent font-medium'
                      : 'border-border bg-surface-secondary text-text-secondary hover:bg-surface-tertiary'
                  }`}
                >
                  {p.name}
                </button>
              ))}
            </div>
          </div>

          {/* API Key */}
          <div>
            <label className="text-xs font-medium text-text-secondary mb-1.5 block">API Key</label>
            <input
              type="password"
              value={apiKey}
              onChange={e => setApiKey(e.target.value)}
              placeholder={current.placeholder}
              className="w-full text-sm px-3 py-2 rounded-lg border border-border bg-surface-secondary text-text-primary placeholder:text-text-tertiary outline-none focus:border-accent transition-colors"
            />
          </div>

          {/* Base URL */}
          <div>
            <label className="text-xs font-medium text-text-secondary mb-1.5 block">Base URL</label>
            <input
              type="text"
              value={baseUrl}
              onChange={e => setBaseUrl(e.target.value)}
              className="w-full text-sm px-3 py-2 rounded-lg border border-border bg-surface-secondary text-text-primary placeholder:text-text-tertiary outline-none focus:border-accent transition-colors font-mono text-xs"
            />
          </div>

          {/* Model */}
          <div>
            <label className="text-xs font-medium text-text-secondary mb-1.5 block">Model</label>
            <select
              value={model}
              onChange={e => setModel(e.target.value)}
              className="w-full text-sm px-3 py-2 rounded-lg border border-border bg-surface-secondary text-text-primary outline-none focus:border-accent transition-colors"
            >
              {current.models.map(m => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-2 px-5 py-3 border-t border-border-light">
          <button
            onClick={onClose}
            className="text-sm px-4 py-1.5 rounded-lg text-text-secondary hover:bg-surface-tertiary transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => {
              // TODO: save settings to backend
              onClose()
            }}
            className="text-sm px-4 py-1.5 rounded-lg bg-accent text-white hover:bg-accent-hover transition-colors"
          >
            Save
          </button>
        </div>
      </div>
    </div>
  )
}