import { useLayoutEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import AnnotationsPanel from './AnnotationsPanel'
import type { Annotation, ChatMessage, PaperFile, PaperSource, RagStep } from '../../types'

interface ChatSidebarProps {
  file: PaperFile
  messages: ChatMessage[]
  onSendMessage: (content: string) => boolean | void
  onQuickAction?: (label: string, prompt: string) => void
  isStreaming?: boolean
  annotations: Annotation[]
  onDeleteAnnotation: (annotationId: string) => void
  onGoToPage: (pageNumber: number) => void
}

const quickActions = [
  { label: '总结全文', prompt: '请基于整篇论文给我一份结构化总结，包括研究问题、方法、结果和局限。' },
  { label: '核心贡献', prompt: '请梳理论文的核心贡献，并指出它相对已有工作的关键差异。' },
  { label: '方法拆解', prompt: '请拆解论文方法部分，按模块说明输入、过程和输出。' },
]

type Tab = 'chat' | 'annotations'

function SourceCard({ source, onGoToPage }: { source: PaperSource; onGoToPage: (pageNumber: number) => void }) {
  return (
    <button
      type="button"
      onClick={() => onGoToPage(source.page_number)}
      className="group overflow-hidden rounded-[20px] border border-border bg-surface px-3 py-3 text-left transition-all hover:border-text-tertiary hover:bg-surface-secondary"
    >
      <div className="flex gap-3">
        {source.asset_url && (
          <img
            src={source.asset_url}
            alt={source.title}
            className="h-16 w-16 shrink-0 rounded-2xl object-cover"
          />
        )}
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="rounded-full bg-surface-secondary px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-text-tertiary">
              {source.type}
            </span>
            <span className="text-xs text-text-tertiary">p.{source.page_number}</span>
          </div>
          <p className="mt-2 text-sm font-medium text-text-primary">
            {source.title}
          </p>
          <p className="mt-1 text-xs leading-5 text-text-secondary">
            {source.content}
          </p>
        </div>
      </div>
    </button>
  )
}

function RagSteps({ steps }: { steps: RagStep[] }) {
  if (!steps.length) return null
  return (
    <div className="rounded-[20px] border border-border bg-surface px-3 py-3">
      <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-text-tertiary">
        Agentic RAG
      </div>
      <div className="space-y-2">
        {steps.slice(-6).map((step, index) => (
          <div key={`${step.label}-${index}`} className="flex gap-2 text-xs leading-5 text-text-secondary">
            <span className="mt-0.5 shrink-0">{step.icon}</span>
            <div>
              <div className="font-medium text-text-primary">{step.label}</div>
              {step.detail && <div>{step.detail}</div>}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function ReferenceBlocks({
  sources,
  onGoToPage,
}: {
  sources: PaperSource[]
  onGoToPage: (pageNumber: number) => void
}) {
  if (!sources.length) return null

  return (
    <details className="group rounded-[20px] border border-border bg-surface px-3 py-3">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-left">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-text-tertiary">
            RAG References
          </div>
          <div className="mt-1 text-sm font-medium text-text-primary">
            参考文本块（{sources.length}）
          </div>
        </div>
        <span className="shrink-0 rounded-full bg-surface-secondary px-2.5 py-1 text-xs text-text-secondary">
          点击展开/收起
        </span>
      </summary>

      <div className="mt-3 grid gap-2 border-t border-border pt-3">
        {sources.map((source) => (
          <SourceCard
            key={source.id}
            source={source}
            onGoToPage={onGoToPage}
          />
        ))}
      </div>
    </details>
  )
}

export default function ChatSidebar({
  file,
  messages,
  onSendMessage,
  onQuickAction,
  isStreaming,
  annotations,
  onDeleteAnnotation,
  onGoToPage,
}: ChatSidebarProps) {
  const [input, setInput] = useState('')
  const [activeTab, setActiveTab] = useState<Tab>('chat')
  const messagesScrollerRef = useRef<HTMLDivElement>(null)
  const previousMessageCountRef = useRef(0)

  useLayoutEffect(() => {
    const scroller = messagesScrollerRef.current
    if (!scroller) return

    const previousCount = previousMessageCountRef.current
    previousMessageCountRef.current = messages.length
    const behavior: ScrollBehavior = messages.length > previousCount ? 'smooth' : 'auto'
    const frame = requestAnimationFrame(() => {
      scroller.scrollTo({ top: scroller.scrollHeight, behavior })
    })
    return () => cancelAnimationFrame(frame)
  }, [messages])

  const submitInput = () => {
    const trimmed = input.trim()
    if (!trimmed || isStreaming) return
    const sent = onSendMessage(trimmed)
    if (sent === false) return
    setInput('')
  }

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    submitInput()
  }

  return (
    <div className="min-h-0 h-full rounded-[28px] border border-border bg-surface shadow-[0_12px_40px_rgba(36,30,20,0.08)]">
      <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-[28px]">
        <div className="border-b border-border bg-[rgba(255,252,245,0.92)] px-4 py-4 backdrop-blur">
          <h2 className="text-lg font-semibold text-text-primary">
            论文问答与阅读笔记
          </h2>

          <div className="mt-4 flex gap-2 rounded-full bg-surface-secondary p-1">
            <button
              onClick={() => setActiveTab('chat')}
              className={`flex-1 rounded-full px-3 py-2 text-sm transition-colors ${
                activeTab === 'chat'
                  ? 'bg-surface text-text-primary shadow-sm'
                  : 'text-text-tertiary hover:text-text-secondary'
              }`}
            >
              对话
            </button>
            <button
              onClick={() => setActiveTab('annotations')}
              className={`flex-1 rounded-full px-3 py-2 text-sm transition-colors ${
                activeTab === 'annotations'
                  ? 'bg-surface text-text-primary shadow-sm'
                  : 'text-text-tertiary hover:text-text-secondary'
              }`}
            >
              笔记 ({annotations.length})
            </button>
          </div>
        </div>

        {activeTab === 'annotations' ? (
          <AnnotationsPanel
            annotations={annotations}
            onDelete={onDeleteAnnotation}
            onGoToPage={onGoToPage}
          />
        ) : (
          <>
            <div
              ref={messagesScrollerRef}
              className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-4 py-4"
            >
              {file.status === 'parsing' || file.status === 'indexing' ? (
                <div className="mb-4 flex items-center gap-2 rounded-[20px] bg-surface-secondary px-4 py-3 text-sm text-text-secondary">
                  <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-accent border-t-transparent" />
                  {file.status === 'parsing' ? '正在解析文档结构…' : '正在构建多模态索引…'}
                </div>
              ) : null}

              <div className="space-y-4">
                {messages.map((msg) => (
                  <div key={msg.id} className={msg.role === 'user' ? 'flex justify-end' : 'flex justify-start'}>
                    <div className={msg.role === 'user'
                      ? 'max-w-[86%] rounded-[24px] rounded-br-md bg-text-primary px-4 py-3 text-sm leading-6 text-white'
                      : 'max-w-[92%] space-y-3'}>
                      {msg.role === 'assistant' ? (
                        <div className="rounded-[24px] rounded-bl-md bg-surface-secondary px-4 py-3 text-sm leading-7 text-text-primary">
                          <ReactMarkdown
                            remarkPlugins={[remarkMath]}
                            rehypePlugins={[rehypeKatex]}
                            components={{
                              p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                              h1: ({ children }) => <h1 className="mb-2 mt-3 text-lg font-bold first:mt-0">{children}</h1>,
                              h2: ({ children }) => <h2 className="mb-2 mt-3 text-base font-bold first:mt-0">{children}</h2>,
                              h3: ({ children }) => <h3 className="mb-1.5 mt-2 text-sm font-bold first:mt-0">{children}</h3>,
                              ul: ({ children }) => <ul className="mb-2 list-disc space-y-1 pl-4 last:mb-0">{children}</ul>,
                              ol: ({ children }) => <ol className="mb-2 list-decimal space-y-1 pl-4 last:mb-0">{children}</ol>,
                              a: ({ children, href }) => (
                                <a href={href} className="text-accent hover:underline" target="_blank" rel="noopener noreferrer">
                                  {children}
                                </a>
                              ),
                              code: ({ children, className }) => {
                                const isBlock = className?.includes('language-')
                                if (isBlock) {
                                  return (
                                    <pre className="my-2 overflow-x-auto rounded-2xl bg-surface px-3 py-3 text-xs font-mono">
                                      <code>{children}</code>
                                    </pre>
                                  )
                                }
                                return (
                                  <code className="rounded bg-surface px-1 py-0.5 text-xs font-mono">
                                    {children}
                                  </code>
                                )
                              },
                              pre: ({ children }) => <div className="overflow-x-auto">{children}</div>,
                            }}
                          >
                            {msg.content}
                          </ReactMarkdown>
                        </div>
                      ) : (
                        msg.content
                      )}

                      {msg.role === 'assistant' && msg.ragSteps && msg.ragSteps.length > 0 && (
                        <RagSteps steps={msg.ragSteps} />
                      )}

                      {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 ? (
                        <ReferenceBlocks
                          sources={msg.sources}
                          onGoToPage={onGoToPage}
                        />
                      ) : null}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="border-t border-border bg-[rgba(255,252,245,0.92)] px-4 py-4 backdrop-blur">
              {file.status === 'ready' && (
                <div className="mb-3 flex gap-2 overflow-x-auto pb-1">
                  {quickActions.map((action) => (
                    <button
                      key={action.label}
                      onClick={() => onQuickAction?.(action.label, action.prompt)}
                      disabled={isStreaming}
                      className="shrink-0 rounded-full border border-border bg-surface px-3 py-1.5 text-xs text-text-secondary transition-colors hover:bg-surface-secondary hover:text-text-primary disabled:opacity-40"
                    >
                      {action.label}
                    </button>
                  ))}
                </div>
              )}

              <form onSubmit={handleSubmit} className="rounded-[24px] border border-border bg-surface px-4 py-3 shadow-sm">
                <textarea
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault()
                      submitInput()
                    }
                  }}
                  placeholder="询问方法细节、实验结论，或让它总结一段内容..."
                  rows={3}
                  className="min-h-[68px] w-full resize-none bg-transparent text-sm leading-6 text-text-primary outline-none placeholder:text-text-tertiary"
                />

                <div className="mt-3 flex items-center justify-between">
                  <span className="text-xs text-text-tertiary">
                    Shift + Enter 换行
                  </span>
                  <button
                    type="submit"
                    disabled={!input.trim() || isStreaming}
                    className="rounded-full bg-accent px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-hover disabled:opacity-40"
                  >
                    发送
                  </button>
                </div>
              </form>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
