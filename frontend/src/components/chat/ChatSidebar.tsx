import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import type { PaperFile, ChatMessage } from '../../types'

interface ChatSidebarProps {
  file: PaperFile
  messages: ChatMessage[]
  onSendMessage: (content: string) => void
  isStreaming?: boolean
}

const quickActions = [
  { label: 'Summarize', prompt: 'Please provide a concise summary of this paper.' },
  { label: 'Key findings', prompt: 'What are the key findings and contributions of this paper?' },
  { label: 'Methodology', prompt: 'Explain the methodology used in this paper.' },
]

export default function ChatSidebar({ file, messages, onSendMessage, isStreaming }: ChatSidebarProps) {
  const [input, setInput] = useState('')
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
      {/* Quick actions */}
      {file.status === 'ready' && messages.length <= 1 && (
        <div className="px-4 pt-3 pb-2 border-b border-border-light shrink-0">
          <p className="text-xs text-text-tertiary mb-2">Quick actions</p>
          <div className="flex flex-wrap gap-1.5">
            {quickActions.map(action => (
              <button
                key={action.label}
                onClick={() => onSendMessage(action.prompt)}
                className="text-xs px-2.5 py-1.5 rounded-md border border-border bg-surface-secondary hover:bg-surface-tertiary text-text-secondary hover:text-text-primary transition-colors"
              >
                {action.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-4 min-h-0">
        {file.status === 'parsing' && (
          <div className="flex items-center gap-2 text-sm text-text-secondary">
            <div className="w-3.5 h-3.5 border-2 border-accent border-t-transparent rounded-full animate-spin" />
            Parsing document...
          </div>
        )}

        {messages.map(msg => (
          <div key={msg.id} className={msg.role === 'user' ? 'flex justify-end' : ''}>
            <div
              className={
                msg.role === 'user'
                  ? 'max-w-[85%] bg-accent text-white rounded-2xl rounded-br-md px-3.5 py-2 text-sm'
                  : 'text-sm text-text-primary leading-relaxed'
              }
            >
              {msg.role === 'assistant' ? (
                <ReactMarkdown
                  remarkPlugins={[remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                  components={{
                    p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                    code: ({ children, className }) => {
                      const isBlock = className?.includes('language-')
                      if (isBlock) {
                        return (
                          <pre className="bg-surface-secondary rounded-md p-3 my-2 overflow-x-auto text-xs font-mono">
                            <code>{children}</code>
                          </pre>
                        )
                      }
                      return (
                        <code className="bg-surface-secondary px-1 py-0.5 rounded text-xs font-mono">
                          {children}
                        </code>
                      )
                    },
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
      <form onSubmit={handleSubmit} className="shrink-0 border-t border-border-light p-3">
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
  )
}