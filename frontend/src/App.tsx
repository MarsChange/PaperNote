import { useState, useCallback, useRef } from 'react'
import { PanelGroup, Panel, PanelResizeHandle } from 'react-resizable-panels'
import PDFViewer from './components/pdf/PDFViewer'
import ChatSidebar from './components/chat/ChatSidebar'
import UploadOverlay from './components/layout/UploadOverlay'
import SettingsModal from './components/settings/SettingsModal'
import { uploadPaper, getPaperStatus, createConversation, streamChat } from './api'
import type { PaperFile, ChatMessage } from './types'

function App() {
  const [file, setFile] = useState<PaperFile | null>(null)
  const [pdfUrl, setPdfUrl] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [showSettings, setShowSettings] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const conversationIdRef = useRef<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const pollStatus = useCallback(async (paperId: string, filename: string) => {
    const poll = async () => {
      try {
        const data = await getPaperStatus(paperId)
        setFile(prev => prev ? { ...prev, status: data.status as PaperFile['status'], summary: data.summary } : null)

        if (data.status === 'ready') {
          // Create conversation
          const conv = await createConversation(paperId)
          conversationIdRef.current = conv.id

          setMessages([{
            id: crypto.randomUUID(),
            role: 'assistant',
            content: `I've finished parsing **${filename}**. Feel free to ask me any questions about this paper.`,
            timestamp: Date.now(),
          }])
          return
        }

        if (data.status === 'error') {
          setMessages([{
            id: crypto.randomUUID(),
            role: 'assistant',
            content: `Failed to parse **${filename}**. Please check that MinerU API is configured correctly in settings.`,
            timestamp: Date.now(),
          }])
          return
        }

        // Still processing — poll again
        setTimeout(poll, 2000)
      } catch {
        setTimeout(poll, 3000)
      }
    }
    poll()
  }, [])

  const handleFileSelect = useCallback(async (selectedFile: File) => {
    const url = URL.createObjectURL(selectedFile)
    setPdfUrl(url)

    const paper: PaperFile = {
      id: crypto.randomUUID(),
      name: selectedFile.name,
      status: 'uploading',
      uploadedAt: Date.now(),
    }
    setFile(paper)
    setMessages([])
    conversationIdRef.current = null

    try {
      const result = await uploadPaper(selectedFile)
      setFile(prev => prev ? { ...prev, id: result.id, status: 'parsing' } : null)
      pollStatus(result.id, selectedFile.name)
    } catch {
      setFile(prev => prev ? { ...prev, status: 'error' } : null)
      setMessages([{
        id: crypto.randomUUID(),
        role: 'assistant',
        content: 'Failed to upload. Is the backend running on port 8000?',
        timestamp: Date.now(),
      }])
    }
  }, [pollStatus])

  const handleSendMessage = useCallback((content: string) => {
    if (!file || !conversationIdRef.current || isStreaming) return

    // Add user message
    const userMsg: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content,
      timestamp: Date.now(),
    }
    setMessages(prev => [...prev, userMsg])

    // Create placeholder for assistant response
    const assistantMsgId = crypto.randomUUID()
    setMessages(prev => [...prev, {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
    }])
    setIsStreaming(true)

    // Stream response
    abortRef.current = streamChat(
      conversationIdRef.current!,
      file.id,
      content,
      {
        onToken: (token) => {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMsgId
                ? { ...m, content: m.content + token }
                : m
            )
          )
        },
        onDone: () => {
          setIsStreaming(false)
        },
        onError: (error) => {
          setMessages(prev =>
            prev.map(m =>
              m.id === assistantMsgId
                ? { ...m, content: m.content || `Error: ${error}` }
                : m
            )
          )
          setIsStreaming(false)
        },
      },
    )
  }, [file, isStreaming])

  if (!file || !pdfUrl) {
    return (
      <>
        <UploadOverlay
          onFileSelect={handleFileSelect}
          onOpenSettings={() => setShowSettings(true)}
        />
        {showSettings && (
          <SettingsModal onClose={() => setShowSettings(false)} />
        )}
      </>
    )
  }

  return (
    <div className="h-full w-full flex flex-col bg-surface">
      <header className="h-11 flex items-center justify-between px-4 border-b border-border-light bg-surface shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-sm font-medium text-text-primary truncate">
            {file.name}
          </span>
          {file.status === 'parsing' && (
            <span className="text-xs text-accent px-1.5 py-0.5 rounded bg-accent-light shrink-0">
              Parsing...
            </span>
          )}
        </div>
        <button
          onClick={() => setShowSettings(true)}
          className="text-text-tertiary hover:text-text-secondary p-1 rounded hover:bg-surface-tertiary transition-colors"
          title="Settings"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
          </svg>
        </button>
      </header>

      <div className="flex-1 min-h-0">
        <PanelGroup direction="horizontal">
          <Panel defaultSize={65} minSize={40}>
            <PDFViewer url={pdfUrl} />
          </Panel>
          <PanelResizeHandle className="w-px bg-border-light hover:bg-accent transition-colors data-[resize-handle-active]:bg-accent" />
          <Panel defaultSize={35} minSize={25}>
            <ChatSidebar
              file={file}
              messages={messages}
              onSendMessage={handleSendMessage}
              isStreaming={isStreaming}
            />
          </Panel>
        </PanelGroup>
      </div>

      {showSettings && (
        <SettingsModal onClose={() => setShowSettings(false)} />
      )}
    </div>
  )
}

export default App