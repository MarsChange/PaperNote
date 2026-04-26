import { useEffect, useState } from 'react'
import { getProviders, getSettings, updateSettings } from '../../api'
import type { LLMProvider } from '../../types'

interface SettingsModalProps {
  onClose: () => void
}

export default function SettingsModal({ onClose }: SettingsModalProps) {
  const [providers, setProviders] = useState<LLMProvider[]>([])
  const [provider, setProvider] = useState('openai')
  const [apiKey, setApiKey] = useState('')
  const [baseUrl, setBaseUrl] = useState('')
  const [model, setModel] = useState('')
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    Promise.all([getProviders(), getSettings()])
      .then(([providerResponse, settings]) => {
        setProviders(providerResponse.providers)
        setProvider(settings.llm_provider || providerResponse.providers[0]?.id || 'openai')

        const selectedProvider = providerResponse.providers.find(item => item.id === (settings.llm_provider || providerResponse.providers[0]?.id))
        setBaseUrl(selectedProvider?.default_base_url || '')
        setModel(settings.llm_model || selectedProvider?.models[0] || '')
      })
      .catch(() => {
        setError('无法加载模型配置')
      })
  }, [])

  const currentProvider = providers.find(item => item.id === provider) || providers[0]

  useEffect(() => {
    if (!currentProvider) return
    setBaseUrl(currentProvider.default_base_url)
    if (!currentProvider.models.includes(model)) {
      setModel(currentProvider.models[0] || '')
    }
  }, [currentProvider])

  const handleSave = async () => {
    setIsSaving(true)
    setError('')
    try {
      await updateSettings({
        llm_provider: provider,
        api_key: apiKey || undefined,
        base_url: baseUrl,
        model,
      })
      onClose()
    } catch {
      setError('保存失败，请检查后端是否可用')
    } finally {
      setIsSaving(false)
    }
  }

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center px-4">
      <div className="absolute inset-0 bg-[rgba(28,22,16,0.22)] backdrop-blur-sm" onClick={onClose} />

      <div className="relative w-full max-w-2xl overflow-hidden rounded-[32px] border border-border bg-[rgba(255,251,244,0.96)] shadow-[0_28px_80px_rgba(36,30,20,0.18)]">
        <div className="border-b border-border px-6 py-5">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-text-tertiary">
                Model Settings
              </p>
              <h2 className="mt-2 text-2xl font-semibold text-text-primary">配置问答模型</h2>
            </div>
            <button
              onClick={onClose}
              className="rounded-full p-2 text-text-tertiary transition-colors hover:bg-surface-secondary hover:text-text-primary"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
          <p className="mt-2 text-sm leading-6 text-text-secondary">
            问答、翻译与论文概述都依赖这里配置的 OpenAI 兼容模型接口。
          </p>
        </div>

        <div className="space-y-5 px-6 py-6">
          <div>
            <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.18em] text-text-tertiary">
              Provider
            </label>
            <div className="grid grid-cols-2 gap-2 md:grid-cols-3">
              {providers.map((item) => (
                <button
                  key={item.id}
                  onClick={() => setProvider(item.id)}
                  className={`rounded-2xl border px-3 py-3 text-sm transition-colors ${
                    provider === item.id
                      ? 'border-accent bg-[rgba(15,118,110,0.08)] text-accent'
                      : 'border-border bg-surface-secondary text-text-secondary hover:bg-surface-tertiary'
                  }`}
                >
                  {item.name}
                </button>
              ))}
            </div>
          </div>

          <div className="grid gap-5 md:grid-cols-2">
            <div>
              <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.18em] text-text-tertiary">
                API Key
              </label>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="在这里输入新的 API Key（留空则保持当前运行时配置）"
                className="w-full rounded-2xl border border-border bg-surface px-4 py-3 text-sm text-text-primary outline-none transition-colors focus:border-accent"
              />
            </div>

            <div>
              <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.18em] text-text-tertiary">
                Model
              </label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="w-full rounded-2xl border border-border bg-surface px-4 py-3 text-sm text-text-primary outline-none transition-colors focus:border-accent"
              >
                {(currentProvider?.models || []).map((item) => (
                  <option key={item} value={item}>{item}</option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <label className="mb-2 block text-xs font-semibold uppercase tracking-[0.18em] text-text-tertiary">
              Base URL
            </label>
            <input
              type="text"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              className="w-full rounded-2xl border border-border bg-surface px-4 py-3 font-mono text-sm text-text-primary outline-none transition-colors focus:border-accent"
            />
          </div>

          {error && (
            <div className="rounded-2xl bg-[rgba(177,70,47,0.1)] px-4 py-3 text-sm text-[#b1462f]">
              {error}
            </div>
          )}
        </div>

        <div className="flex items-center justify-between border-t border-border bg-[rgba(255,252,245,0.92)] px-6 py-4">
          <button
            onClick={onClose}
            className="text-sm text-text-tertiary transition-colors hover:text-text-secondary"
          >
            取消
          </button>
          <button
            onClick={handleSave}
            disabled={isSaving}
            className="rounded-full bg-accent px-5 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-hover disabled:opacity-40"
          >
            {isSaving ? '保存中…' : '保存配置'}
          </button>
        </div>
      </div>
    </div>
  )
}
