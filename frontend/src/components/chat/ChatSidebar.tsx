import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import AnnotationsPanel from './AnnotationsPanel'
import type { PaperFile, ChatMessage, Annotation } from '../../types'

interface ChatSidebarProps {
  file: PaperFile
  messages: ChatMessage[]
  onSendMessage: (content: string) => void
  onQuickAction?: (label: string, prompt: string) => void
  isStreaming?: boolean
  annotations: Annotation[]
  onDeleteAnnotation: (annotationId: string) => void
  onGoToPage: (pageNumber: number) => void
}

const quickActions = [
  { label: 'Summarize', prompt: 'Please provide a concise summary of this paper.' },
  { label: 'Key findings', prompt: 'What are the key findings and contributions of this paper?' },
  { label: 'Methodology', prompt: 'Explain the methodology used in this paper.' },
]

type Tab = 'chat' | 'annotations'

export default function ChatSidebar({ file, messages, onSendMessage, onQuickAction, isStreaming, annotations, onDeleteAnnotation, onGoToPage }: ChatSidebarProps) {
  const [input, setInput] = useState('')
  const [activeTab, setActiveTab] = useState<Tab>('chat')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    const trimmed = input.trim()
    if (!trimmed) return
    onSendMessage(trimmed)
    setInput('')
  }

  return (
    <div className="h-full flex flex-col bg-surface">
      {/* Tab switcher */}
      <div className="flex border-b border-border-light shrink-0">
        <button
          onClick={() => setActiveTab('chat')}
          className={`flex-1 text-xs py-2 font-medium transition-colors ${
            activeTab === 'chat'
              ? 'text-text-primary border-b-2 border-accent'
              : 'text-text-tertiary hover:text-text-secondary'
          }`}
        >
          Chat
        </button>
        <button
          onClick={() => setActiveTab('annotations')}
          className={`flex-1 text-xs py-2 font-medium transition-colors ${
            activeTab === 'annotations'
              ? 'text-text-primary border-b-2 border-accent'
              : 'text-text-tertiary hover:text-text-secondary'
          }`}
        >
          Annotations
          {annotations.length > 0 && (
            <span className="ml-1 text-text-tertiary">({annotations.length})</span>
          )}
        </button>
      </div>

      {activeTab === 'annotations' ? (
        <AnnotationsPanel
          annotations={annotations}
          onDelete={onDeleteAnnotation}
          onGoToPage={onGoToPage}
        />
      ) : (
        <>
          {/* Messages */}
          <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4 min-h-0">
            {file.status === 'parsing' && (
              <div className="flex items-center gap-2 text-sm text-text-secondary">
                <div className="w-3.5 h-3.5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
                Parsing document...
              </div>
            )}

            {messages.map(msg => (
              <div key={msg.id} className={msg.role === 'user' ? 'flex justify-end' : 'flex justify-start'}>
                <div
                  className={
                    msg.role === 'user'
                      ? 'max-w-[85%] bg-accent text-white rounded-2xl rounded-br-md px-3.5 py-2 text-sm'
                      : 'max-w-[90%] bg-surface-secondary text-text-primary rounded-2xl rounded-bl-md px-3.5 py-2.5 text-sm leading-relaxed'
                  }
                >
                  {msg.role === 'assistant' ? (
                    <ReactMarkdown
                      remarkPlugins={[remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                      components={{
                        p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                        h1: ({ children }) => <h1 className="text-lg font-bold mb-2 mt-3 first:mt-0">{children}</h1>,
                        h2: ({ children }) => <h2 className="text-base font-bold mb-2 mt-3 first:mt-0">{children}</h2>,
                        h3: ({ children }) => <h3 className="text-sm font-bold mb-1.5 mt-2 first:mt-0">{children}</h3>,
                        ul: ({ children }) => <ul className="pl-4 list-disc mb-2 last:mb-0 space-y-0.5">{children}</ul>,
                        ol: ({ children }) => <ol className="pl-4 list-decimal mb-2 last:mb-0 space-y-0.5">{children}</ol>,
                        li: ({ children }) => <li className="text-sm">{children}</li>,
                        a: ({ children, href }) => (
                          <a href={href} className="text-accent hover:underline" target="_blank" rel="noopener noreferrer">
                            {children}
                          </a>
                        ),
                        code: ({ children, className }) => {
                          const isBlock = className?.includes('language-')
                          if (isBlock) {
                            return (
                              <pre className="bg-surface-tertiary rounded-md p-3 my-2 overflow-x-auto text-xs font-mono">
                                <code>{children}</code>
                              </pre>
                            )
                          }
                          return (
                            <code className="bg-surface-tertiary px-1 py-0.5 rounded text-xs font-mono">
                              {children}
                            </code>
                          )
                        },
                        pre: ({ children }) => <div className="overflow-x-auto max-w-full">{children}</div>,
                      }}
                    >
                      {msg.content}
                    </ReactMarkdown>
                  ) : (
                    msg.content
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="shrink-0 border-t border-border-light">
            {file.status === 'ready' && (
              <div className="flex gap-1.5 px-3 pt-2 overflow-x-auto">
                {quickActions.map(action => (
                  <button
                    key={action.label}
                    onClick={() => onQuickAction?.(action.label, action.prompt)}
                    disabled={isStreaming}
                    className="text-xs px-2 py-1 rounded-md border border-border bg-surface-secondary hover:bg-surface-tertiary text-text-secondary hover:text-text-primary transition-colors whitespace-nowrap shrink-0 disabled:opacity-40"
                  >
                    {action.label}
                  </button>
                ))}
              </div>
            )}
            <form onSubmit={handleSubmit} className="p-3 pt-2">
            <div className="flex items-end gap-2 bg-surface-secondary rounded-xl px-3 py-2">
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault()
                    handleSubmit(e)
                  }
                }}
                placeholder="Ask about this paper..."
                rows={1}
                className="flex-1 bg-transparent text-sm text-text-primary placeholder:text-text-tertiary resize-none outline-none max-h-32"
              />
              <button
                type="submit"
                disabled={!input.trim() || isStreaming}
                className="p-1.5 rounded-lg bg-accent text-white disabled:opacity-30 hover:bg-accent-hover transition-colors shrink-0"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M5 12h14M12 5l7 7-7 7" />
                </svg>
              </button>
            </div>
          </form>
          </div>
        </>
      )}
    </div>
  )
}
