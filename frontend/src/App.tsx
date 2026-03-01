import { useState, useCallback, useRef } from 'react'
import { PanelGroup, Panel, PanelResizeHandle } from 'react-resizable-panels'
import PDFViewer from './components/pdf/PDFViewer'
import ChatSidebar from './components/chat/ChatSidebar'
import UploadOverlay from './components/layout/UploadOverlay'
import SettingsModal from './components/settings/SettingsModal'
import HistorySidebar from './components/layout/HistorySidebar'
import { uploadPaper, getPaperStatus, createConversation, streamChat, fetchAnnotations, createAnnotation, deleteAnnotation, fetchPaperDetail, fetchMessages } from './api'
import type { PaperFile, ChatMessage, Annotation } from './types'

function App() {
  const [file, setFile] = useState<PaperFile | null>(null)
  const [pdfUrl, setPdfUrl] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [showSettings, setShowSettings] = useState(false)
  const [showHistory, setShowHistory] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [scrollToPage, setScrollToPage] = useState(0)
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

    // Add user message + assistant placeholder atomically
    const assistantMsgId = crypto.randomUUID()
    setMessages(prev => [
      ...prev,
      { id: crypto.randomUUID(), role: 'user' as const, content, timestamp: Date.now() },
      { id: assistantMsgId, role: 'assistant' as const, content: '', timestamp: Date.now() },
    ])
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

  const handleQuickAction = useCallback((_label: string, prompt: string) => {
    if (!file || !conversationIdRef.current || isStreaming) return

    // No user bubble for quick actions — just create assistant placeholder directly
    const assistantMsgId = crypto.randomUUID()
    setMessages(prev => [
      ...prev,
      { id: assistantMsgId, role: 'assistant' as const, content: '', timestamp: Date.now() },
    ])
    setIsStreaming(true)

    // Stream with prompt
    abortRef.current = streamChat(
      conversationIdRef.current!,
      file.id,
      prompt,
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

  const loadAnnotations = useCallback(async (paperId: string) => {
    try {
      const data = await fetchAnnotations(paperId)
      setAnnotations(data)
    } catch {
      setAnnotations([])
    }
  }, [])

  const handleDeleteAnnotation = useCallback(async (annotationId: string) => {
    if (!file) return
    try {
      await deleteAnnotation(file.id, annotationId)
      setAnnotations(prev => prev.filter(a => a.id !== annotationId))
    } catch {
      // silently fail
    }
  }, [file])

  const handleCreateAnnotation = useCallback(async (color: string, text: string, pageNumber: number) => {
    if (!file) return
    const normalizedText = text.replace(/\s+/g, ' ').trim()
    if (!normalizedText) return
    try {
      const ann = await createAnnotation(file.id, {
        page_number: pageNumber,
        text_content: normalizedText,
        color,
      })
      setAnnotations(prev => [...prev, ann])
    } catch (err) {
      console.error('Failed to create annotation:', err)
    }
  }, [file])

  const handleSelectPaper = useCallback(async (paper: { id: string; filename: string }) => {
    const url = `/api/papers/${paper.id}/pdf`
    setPdfUrl(url)
    setFile({
      id: paper.id,
      name: paper.filename,
      status: 'ready',
      uploadedAt: Date.now(),
    })
    setMessages([])
    conversationIdRef.current = null
    setAnnotations([])

    try {
      // Check for existing conversations
      const detail = await fetchPaperDetail(paper.id)
      const existingConversations = detail.conversations || []

      let convId: string

      if (existingConversations.length > 0) {
        // Use the most recent conversation (already sorted DESC by backend)
        convId = existingConversations[0].id
        conversationIdRef.current = convId

        // Load existing messages
        const msgs = await fetchMessages(convId)
        if (msgs.length > 0) {
          setMessages(msgs.map(m => ({
            id: m.id,
            role: m.role as 'user' | 'assistant',
            content: m.content,
            timestamp: new Date(m.created_at).getTime(),
          })))
        } else {
          setMessages([{
            id: crypto.randomUUID(),
            role: 'assistant',
            content: `Loaded **${paper.filename}**. Feel free to ask me any questions about this paper.`,
            timestamp: Date.now(),
          }])
        }
      } else {
        // No existing conversation — create a new one
        const conv = await createConversation(paper.id)
        convId = conv.id
        conversationIdRef.current = convId
        setMessages([{
          id: crypto.randomUUID(),
          role: 'assistant',
          content: `Loaded **${paper.filename}**. Feel free to ask me any questions about this paper.`,
          timestamp: Date.now(),
        }])
      }

      loadAnnotations(paper.id)
    } catch {
      setMessages([{
        id: crypto.randomUUID(),
        role: 'assistant',
        content: 'Failed to load conversation for this paper.',
        timestamp: Date.now(),
      }])
    }
  }, [loadAnnotations])

  const handleNewPaper = useCallback(() => {
    setFile(null)
    setPdfUrl(null)
    setMessages([])
    setAnnotations([])
    conversationIdRef.current = null
    abortRef.current?.abort()
    abortRef.current = null
  }, [])

  if (!file || !pdfUrl) {
    return (
      <>
        <UploadOverlay
          onFileSelect={handleFileSelect}
          onOpenSettings={() => setShowSettings(true)}
          onOpenHistory={() => setShowHistory(true)}
        />
        <HistorySidebar
          open={showHistory}
          onClose={() => setShowHistory(false)}
          onSelectPaper={handleSelectPaper}
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
          <button
            onClick={() => setShowHistory(true)}
            className="group relative text-text-tertiary hover:text-text-secondary p-1 rounded hover:bg-surface-tertiary transition-colors shrink-0"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 12h18M3 6h18M3 18h18" />
            </svg>
            <span className="pointer-events-none absolute top-full left-1/2 -translate-x-1/2 mt-1 whitespace-nowrap rounded bg-text-primary px-2 py-0.5 text-xs text-surface opacity-0 group-hover:opacity-100 transition-opacity z-10">
              Papers
            </span>
          </button>
          <button
            onClick={handleNewPaper}
            className="group relative flex items-center gap-1 px-2 py-0.5 text-xs text-text-secondary rounded-md border border-border bg-surface hover:bg-surface-secondary shadow-sm hover:shadow transition-all shrink-0"
          >
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            New
            <span className="pointer-events-none absolute top-full left-1/2 -translate-x-1/2 mt-1 whitespace-nowrap rounded bg-text-primary px-2 py-0.5 text-xs text-surface opacity-0 group-hover:opacity-100 transition-opacity z-10">
              Upload new paper
            </span>
          </button>
          <span className="text-sm font-medium text-text-primary truncate">
            {file.name}
          </span>
          {file.status === 'parsing' && (
            <span className="text-xs text-accent px-1.5 py-0.5 rounded bg-accent-light shrink-0">
              Parsing...
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setShowSettings(true)}
            className="group relative text-text-tertiary hover:text-text-secondary p-1 rounded hover:bg-surface-tertiary transition-colors"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
            </svg>
            <span className="pointer-events-none absolute top-full right-0 mt-1 whitespace-nowrap rounded bg-text-primary px-2 py-0.5 text-xs text-surface opacity-0 group-hover:opacity-100 transition-opacity z-10">
              Settings
            </span>
          </button>
        </div>
      </header>

      <div className="flex-1 min-h-0">
        <PanelGroup direction="horizontal">
          <Panel defaultSize={65} minSize={40}>
            <PDFViewer
              url={pdfUrl}
              annotations={annotations}
              onAnnotation={handleCreateAnnotation}
              scrollToPage={scrollToPage}
            />
          </Panel>
          <PanelResizeHandle className="w-px bg-border-light hover:bg-accent transition-colors data-[resize-handle-active]:bg-accent" />
          <Panel defaultSize={35} minSize={25}>
            <ChatSidebar
              file={file}
              messages={messages}
              onSendMessage={handleSendMessage}
              onQuickAction={handleQuickAction}
              isStreaming={isStreaming}
              annotations={annotations}
              onDeleteAnnotation={handleDeleteAnnotation}
              onGoToPage={(page: number) => setScrollToPage(page)}
            />
          </Panel>
        </PanelGroup>
      </div>

      <HistorySidebar
        open={showHistory}
        onClose={() => setShowHistory(false)}
        onSelectPaper={handleSelectPaper}
        currentPaperId={file.id}
      />

      {showSettings && (
        <SettingsModal onClose={() => setShowSettings(false)} />
      )}
    </div>
  )
}

export default App
