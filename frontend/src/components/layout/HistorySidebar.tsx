import { useEffect, useState } from 'react'
import { deletePaper, fetchPapers } from '../../api'

interface Paper {
  id: string
  filename: string
  title?: string
  status: string
  summary?: string
  created_at: string
}

interface HistorySidebarProps {
  open: boolean
  onClose: () => void
  onSelectPaper: (paper: Paper) => void
  currentPaperId?: string
}

export default function HistorySidebar({ open, onClose, onSelectPaper, currentPaperId }: HistorySidebarProps) {
  const [papers, setPapers] = useState<Paper[]>([])
  const [confirmDeleteId, setConfirmDeleteId] = useState<string | null>(null)

  useEffect(() => {
    if (!open) return
    fetchPapers()
      .then((data) => setPapers(data))
      .catch(() => setPapers([]))
  }, [open])

  useEffect(() => {
    if (!open) return
    const handleKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [open, onClose])

  const handleDelete = async (paperId: string) => {
    try {
      await deletePaper(paperId)
      setPapers((prev) => prev.filter((paper) => paper.id !== paperId))
    } catch {
      // noop
    }
    setConfirmDeleteId(null)
  }

  return (
    <>
      {open && (
        <div
          className="fixed inset-0 z-40 bg-[rgba(28,22,16,0.22)] backdrop-blur-sm"
          onClick={onClose}
        />
      )}

      <aside
        className={`fixed left-0 top-0 z-50 flex h-full w-[360px] max-w-[92vw] flex-col border-r border-border bg-[rgba(255,251,244,0.94)] shadow-[0_30px_70px_rgba(36,30,20,0.16)] backdrop-blur transition-transform duration-200 ${
          open ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="border-b border-border px-5 py-5">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-text-tertiary">
                Library
              </p>
              <h2 className="mt-2 text-xl font-semibold text-text-primary">论文文库</h2>
            </div>
            <button
              onClick={onClose}
              className="rounded-full p-2 text-text-tertiary transition-colors hover:bg-surface-secondary hover:text-text-primary"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M18 6L6 18M6 6l12 12" />
              </svg>
            </button>
          </div>
          <p className="mt-2 text-sm leading-6 text-text-secondary">
            重新打开已解析的论文，延续问答和阅读笔记。
          </p>
        </div>

        <div className="flex-1 overflow-y-auto px-4 py-4">
          {papers.length === 0 ? (
            <div className="rounded-[24px] border border-border bg-surface-secondary px-5 py-6 text-center">
              <p className="text-base font-medium text-text-primary">还没有论文记录</p>
              <p className="mt-2 text-sm leading-6 text-text-tertiary">
                上传第一篇论文后，它会出现在这里。
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {papers.map((paper) => {
                const displayTitle = paper.title?.trim() || paper.filename
                return (
                <div
                  key={paper.id}
                  className={`group rounded-[24px] border px-4 py-4 transition-all ${
                    paper.id === currentPaperId
                      ? 'border-accent bg-[rgba(15,118,110,0.08)]'
                      : 'border-border bg-surface hover:border-text-tertiary hover:bg-surface-secondary'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <button
                      type="button"
                      onClick={() => {
                        onSelectPaper(paper)
                        onClose()
                      }}
                      className="min-w-0 flex-1 text-left"
                    >
                      <div className="flex items-center gap-2">
                        <span className="rounded-full bg-surface-secondary px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-text-tertiary">
                          {paper.status}
                        </span>
                        <span className="text-xs text-text-tertiary">
                          {new Date(paper.created_at).toLocaleString()}
                        </span>
                      </div>
                      <p className="mt-3 text-base font-medium text-text-primary">
                        {displayTitle}
                      </p>
                      {displayTitle !== paper.filename && (
                        <p className="mt-1 truncate text-xs text-text-tertiary">
                          {paper.filename}
                        </p>
                      )}
                      {paper.summary && (
                        <p className="mt-2 text-sm leading-6 text-text-secondary">
                          {paper.summary}
                        </p>
                      )}
                    </button>

                    {confirmDeleteId === paper.id ? (
                      <div className="flex shrink-0 flex-col gap-2">
                        <button
                          type="button"
                          onClick={() => handleDelete(paper.id)}
                          className="rounded-full bg-[#b1462f] px-3 py-1.5 text-xs font-medium text-white"
                        >
                          删除
                        </button>
                        <button
                          type="button"
                          onClick={() => setConfirmDeleteId(null)}
                          className="rounded-full border border-border px-3 py-1.5 text-xs text-text-secondary"
                        >
                          取消
                        </button>
                      </div>
                    ) : (
                      <button
                        type="button"
                        onClick={() => setConfirmDeleteId(paper.id)}
                        className="shrink-0 rounded-full p-2 text-text-tertiary opacity-0 transition-all hover:bg-surface-secondary hover:text-[#b1462f] group-hover:opacity-100"
                      >
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M18 6L6 18M6 6l12 12" />
                        </svg>
                      </button>
                    )}
                  </div>
                </div>
                )
              })}
            </div>
          )}
        </div>
      </aside>
    </>
  )
}
