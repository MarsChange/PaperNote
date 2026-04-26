import { useCallback, useEffect, useRef, useState } from 'react'
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels'
import PDFViewer from './components/pdf/PDFViewer'
import ChatSidebar from './components/chat/ChatSidebar'
import UploadOverlay from './components/layout/UploadOverlay'
import SettingsModal from './components/settings/SettingsModal'
import HistorySidebar from './components/layout/HistorySidebar'
import {
  createAnnotation,
  createConversation,
  deleteAnnotation,
  fetchAnnotations,
  fetchMessages,
  fetchPaperDetail,
  getPaperStatus,
  streamChat,
  translateSelection,
  uploadPaper,
} from './api'
import type { Annotation, ChatMessage, PaperFile, PaperSource } from './types'

function parseSources(metadataJson?: string): PaperSource[] {
  if (!metadataJson) return []
  try {
    const parsed = JSON.parse(metadataJson)
    return Array.isArray(parsed?.sources) ? parsed.sources : []
  } catch {
    return []
  }
}

function fallbackTitleFromFilename(filename: string) {
  return filename.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').trim() || filename
}

function App() {
  const [file, setFile] = useState<PaperFile | null>(null)
  const [pdfUrl, setPdfUrl] = useState<string | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [showSettings, setShowSettings] = useState(false)
  const [showHistory, setShowHistory] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [scrollToPage, setScrollToPage] = useState(0)
  const [isCompact, setIsCompact] = useState(false)

  const conversationIdRef = useRef<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  useEffect(() => {
    const media = window.matchMedia('(max-width: 1024px)')
    const handleChange = () => setIsCompact(media.matches)
    handleChange()
    media.addEventListener('change', handleChange)
    return () => media.removeEventListener('change', handleChange)
  }, [])

  const loadAnnotations = useCallback(async (paperId: string) => {
    try {
      const data = await fetchAnnotations(paperId)
      setAnnotations(data)
    } catch {
      setAnnotations([])
    }
  }, [])

  const pollStatus = useCallback(async (paperId: string, fallbackName: string) => {
    const poll = async () => {
      try {
        const data = await getPaperStatus(paperId)
        const displayName = data.title?.trim() || fallbackName
        setFile((prev) => prev ? {
          ...prev,
          name: displayName,
          filename: data.filename || prev.filename,
          status: data.status as PaperFile['status'],
          summary: data.summary,
          keywords: data.keywords || [],
          pageCount: data.page_count,
          metadata: data.metadata,
        } : null)

        if (data.status === 'ready') {
          const conv = await createConversation(paperId)
          conversationIdRef.current = conv.id
          await loadAnnotations(paperId)
          setMessages([
            {
              id: crypto.randomUUID(),
              role: 'assistant',
              content: `**${displayName}** 已完成解析。你现在可以询问方法细节、实验结果、图表含义，或者直接让系统总结全文。`,
              timestamp: Date.now(),
            },
          ])
          return
        }

        if (data.status === 'error') {
          setMessages([
            {
              id: crypto.randomUUID(),
              role: 'assistant',
              content: '文档解析失败。请检查 MinerU 配置或模型设置后重试。',
              timestamp: Date.now(),
            },
          ])
          return
        }

        setTimeout(poll, data.status === 'indexing' ? 2500 : 2000)
      } catch {
        setTimeout(poll, 3000)
      }
    }

    poll()
  }, [loadAnnotations])

  const handleFileSelect = useCallback(async (selectedFile: File) => {
    const url = URL.createObjectURL(selectedFile)
    setPdfUrl(url)
    setAnnotations([])
    setMessages([])
    conversationIdRef.current = null

    setFile({
      id: crypto.randomUUID(),
      name: fallbackTitleFromFilename(selectedFile.name),
      filename: selectedFile.name,
      status: 'uploading',
      uploadedAt: Date.now(),
    })

    try {
      const result = await uploadPaper(selectedFile)
      setFile((prev) => prev ? {
        ...prev,
        id: result.id,
        name: result.title || prev.name,
        filename: result.filename,
        status: 'parsing',
      } : null)
      await pollStatus(result.id, result.title || fallbackTitleFromFilename(selectedFile.name))
    } catch {
      setFile((prev) => prev ? { ...prev, status: 'error' } : null)
      setMessages([
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: '上传失败。请确认后端服务是否已启动。',
          timestamp: Date.now(),
        },
      ])
    }
  }, [pollStatus])

  const handleSendMessage = useCallback((content: string) => {
    if (!file || !conversationIdRef.current || isStreaming) return

    const assistantMsgId = crypto.randomUUID()
    setMessages((prev) => [
      ...prev,
      { id: crypto.randomUUID(), role: 'user', content, timestamp: Date.now() },
      { id: assistantMsgId, role: 'assistant', content: '', timestamp: Date.now(), sources: [] },
    ])
    setIsStreaming(true)

    abortRef.current = streamChat(
      conversationIdRef.current,
      file.id,
      content,
      {
        onToken: (token) => {
          setMessages((prev) => prev.map((message) => (
            message.id === assistantMsgId
              ? { ...message, content: message.content + token }
              : message
          )))
        },
        onDone: (_answer, sources) => {
          setMessages((prev) => prev.map((message) => (
            message.id === assistantMsgId
              ? { ...message, sources }
              : message
          )))
          setIsStreaming(false)
        },
        onError: (error) => {
          setMessages((prev) => prev.map((message) => (
            message.id === assistantMsgId
              ? { ...message, content: message.content || `Error: ${error}` }
              : message
          )))
          setIsStreaming(false)
        },
      },
    )
  }, [file, isStreaming])

  const handleQuickAction = useCallback((label: string, prompt: string) => {
    if (!file || !conversationIdRef.current || isStreaming) return

    const assistantMsgId = crypto.randomUUID()
    setMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role: 'user',
        content: label,
        timestamp: Date.now(),
      },
      {
        id: assistantMsgId,
        role: 'assistant',
        content: '',
        timestamp: Date.now(),
        sources: [],
      },
    ])
    setIsStreaming(true)

    abortRef.current = streamChat(
      conversationIdRef.current,
      file.id,
      prompt,
      {
        onToken: (token) => {
          setMessages((prev) => prev.map((message) => (
            message.id === assistantMsgId
              ? { ...message, content: message.content + token }
              : message
          )))
        },
        onDone: (_answer, sources) => {
          setMessages((prev) => prev.map((message) => (
            message.id === assistantMsgId
              ? { ...message, sources }
              : message
          )))
          setIsStreaming(false)
        },
        onError: (error) => {
          setMessages((prev) => prev.map((message) => (
            message.id === assistantMsgId
              ? { ...message, content: message.content || `Error: ${error}` }
              : message
          )))
          setIsStreaming(false)
        },
      },
    )
  }, [file, isStreaming])

  const handleDeleteAnnotation = useCallback(async (annotationId: string) => {
    if (!file) return
    try {
      await deleteAnnotation(file.id, annotationId)
      setAnnotations((prev) => prev.filter((annotation) => annotation.id !== annotationId))
    } catch {
      // noop
    }
  }, [file])

  const handleCreateAnnotation = useCallback(async (
    color: string,
    text: string,
    pageNumber: number,
    note?: string,
  ) => {
    if (!file) return
    const normalizedText = text.replace(/\s+/g, ' ').trim()
    if (!normalizedText) return

    try {
      const annotation = await createAnnotation(file.id, {
        page_number: pageNumber,
        text_content: normalizedText,
        color,
        note,
      })
      setAnnotations((prev) => [...prev, annotation])
    } catch (error) {
      console.error('Failed to create annotation:', error)
    }
  }, [file])

  const handleTranslateSelection = useCallback(async (text: string, _pageNumber: number) => {
    if (!file) return ''
    return translateSelection(file.id, text)
  }, [file])

  const handleSelectPaper = useCallback(async (paper: { id: string; filename: string; title?: string }) => {
    setPdfUrl(`/api/papers/${paper.id}/pdf`)
    setAnnotations([])
    setMessages([])
    conversationIdRef.current = null

    try {
      const detail = await fetchPaperDetail(paper.id)
      setFile({
        id: paper.id,
        name: detail.title || paper.title || fallbackTitleFromFilename(detail.filename),
        filename: detail.filename,
        status: detail.status as PaperFile['status'],
        summary: detail.summary,
        keywords: detail.keywords || [],
        pageCount: detail.page_count,
        metadata: detail.metadata,
        uploadedAt: Date.now(),
      })

      await loadAnnotations(paper.id)

      const existingConversations = detail.conversations || []
      if (existingConversations.length > 0) {
        const convId = existingConversations[0].id
        conversationIdRef.current = convId
        const storedMessages = await fetchMessages(convId)
        if (storedMessages.length > 0) {
          setMessages(storedMessages.map((message) => ({
            id: message.id,
            role: message.role as ChatMessage['role'],
            content: message.content,
            sources: parseSources(message.metadata_json),
            timestamp: new Date(message.created_at).getTime(),
          })))
        } else {
          setMessages([
            {
              id: crypto.randomUUID(),
              role: 'assistant',
              content: `已载入 **${detail.title || detail.filename}**。你可以继续追问，也可以跳到右侧让系统重新总结全文。`,
              timestamp: Date.now(),
            },
          ])
        }
      } else {
        const conv = await createConversation(paper.id)
        conversationIdRef.current = conv.id
        setMessages([
          {
            id: crypto.randomUUID(),
            role: 'assistant',
            content: `已载入 **${detail.title || detail.filename}**。从任意问题开始即可。`,
            timestamp: Date.now(),
          },
        ])
      }
    } catch {
      setMessages([
        {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: '加载论文会话失败，请稍后重试。',
          timestamp: Date.now(),
        },
      ])
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
        {showSettings && <SettingsModal onClose={() => setShowSettings(false)} />}
      </>
    )
  }

  return (
    <div className="workspace-shell flex h-full w-full flex-col overflow-hidden px-3 py-3 md:px-5 md:py-5">
      <header className="workspace-header mb-3 rounded-[32px] border border-border bg-[rgba(255,251,244,0.86)] px-5 py-4 shadow-[0_20px_60px_rgba(36,30,20,0.08)] backdrop-blur md:px-6">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-full bg-surface-secondary px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-text-tertiary">
                {file.status}
              </span>
              {file.pageCount ? (
                <span className="rounded-full bg-surface-secondary px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-text-tertiary">
                  {file.pageCount} Pages
                </span>
              ) : null}
            </div>
            <h1 className="mt-3 truncate font-display text-2xl text-text-primary md:text-4xl">
              {file.name}
            </h1>
            {file.filename && file.filename !== file.name && (
              <p className="mt-1 truncate text-xs text-text-tertiary md:text-sm">
                源文件：{file.filename}
              </p>
            )}
            {file.summary && (
              <p className="mt-3 max-w-4xl text-sm leading-7 text-text-secondary md:text-base">
                {file.summary}
              </p>
            )}
            {file.keywords && file.keywords.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-2">
                {file.keywords.map((keyword) => (
                  <span
                    key={keyword}
                    className="rounded-full border border-border bg-surface px-3 py-1 text-xs text-text-secondary"
                  >
                    {keyword}
                  </span>
                ))}
              </div>
            )}
          </div>

          <div className="flex shrink-0 flex-wrap gap-2">
            <button
              onClick={() => setShowHistory(true)}
              className="rounded-full border border-border bg-surface px-4 py-2 text-sm text-text-secondary transition-colors hover:bg-surface-secondary"
            >
              文库历史
            </button>
            <button
              onClick={handleNewPaper}
              className="rounded-full border border-border bg-surface px-4 py-2 text-sm text-text-secondary transition-colors hover:bg-surface-secondary"
            >
              新建阅读
            </button>
            <button
              onClick={() => setShowSettings(true)}
              className="rounded-full bg-text-primary px-4 py-2 text-sm font-medium text-surface transition-colors hover:bg-[#30271e]"
            >
              模型设置
            </button>
          </div>
        </div>
      </header>

      <div className="min-h-0 flex-1">
        <PanelGroup direction={isCompact ? 'vertical' : 'horizontal'}>
          <Panel defaultSize={isCompact ? 58 : 62} minSize={30}>
            <PDFViewer
              url={pdfUrl}
              annotations={annotations}
              onAnnotation={handleCreateAnnotation}
              onTranslateSelection={handleTranslateSelection}
              scrollToPage={scrollToPage}
            />
          </Panel>

          <PanelResizeHandle className={isCompact ? 'flex h-3 items-center justify-center' : 'flex w-3 items-center justify-center'}>
            <div className={isCompact ? 'h-px w-16 rounded-full bg-border' : 'h-16 w-px rounded-full bg-border'} />
          </PanelResizeHandle>

          <Panel defaultSize={isCompact ? 42 : 38} minSize={25}>
            <ChatSidebar
              file={file}
              messages={messages}
              onSendMessage={handleSendMessage}
              onQuickAction={handleQuickAction}
              isStreaming={isStreaming}
              annotations={annotations}
              onDeleteAnnotation={handleDeleteAnnotation}
              onGoToPage={(page) => setScrollToPage(page)}
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

      {showSettings && <SettingsModal onClose={() => setShowSettings(false)} />}
    </div>
  )
}

export default App
