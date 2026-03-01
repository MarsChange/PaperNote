import { useState, useEffect, useRef } from 'react'
import { fetchPapers, deletePaper } from '../../api'

interface Paper {
  id: string
  filename: string
  status: string
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
  const panelRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (open) {
      fetchPapers()
        .then((data: Paper[]) => {
          const sorted = [...data].sort(
            (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
          )
          setPapers(sorted)
        })
        .catch(() => setPapers([]))
    }
  }, [open])

  useEffect(() => {
    if (!open) return
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', handleKey)
    return () => document.removeEventListener('keydown', handleKey)
  }, [open, onClose])

  const handleDelete = async (paperId: string) => {
    try {
      await deletePaper(paperId)
      setPapers(prev => prev.filter(p => p.id !== paperId))
    } catch {
      // silently fail
    }
    setConfirmDeleteId(null)
  }

  const formatDate = (dateStr: string) => {
    const d = new Date(dateStr)
    const year = d.getFullYear()
    const month = String(d.getMonth() + 1).padStart(2, '0')
    const day = String(d.getDate()).padStart(2, '0')
    const hours = String(d.getHours()).padStart(2, '0')
    const minutes = String(d.getMinutes()).padStart(2, '0')
    return `${year}-${month}-${day} ${hours}:${minutes}`
  }

  const truncate = (str: string, max: number) =>
    str.length > max ? str.slice(0, max) + '...' : str

  return (
    <>
      {/* Backdrop */}
      {open && (
        <div
          className="fixed inset-0 z-30 bg-black/20"
          onClick={onClose}
        />
      )}

      {/* Panel */}
      <div
        ref={panelRef}
        className={`fixed top-0 left-0 h-full w-72 z-40 bg-surface shadow-lg border-r border-border-light flex flex-col transition-transform duration-200 ${
          open ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Header */}
        <div className="h-11 flex items-center justify-between px-4 border-b border-border-light shrink-0">
          <span className="text-sm font-medium text-text-primary">Papers</span>
          <button
            onClick={onClose}
            className="text-text-tertiary hover:text-text-secondary p-1 rounded hover:bg-surface-tertiary transition-colors"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* List */}
        <div className="flex-1 overflow-y-auto">
          {papers.length === 0 ? (
            <p className="text-xs text-text-tertiary px-4 py-6 text-center">
              No papers yet
            </p>
          ) : (
            papers.map(paper => (
              <div
                key={paper.id}
                className={`group flex items-center gap-2 px-4 py-2.5 cursor-pointer hover:bg-surface-secondary transition-colors ${
                  paper.id === currentPaperId ? 'bg-surface-secondary' : ''
                }`}
                onClick={() => {
                  onSelectPaper(paper)
                  onClose()
                }}
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-text-primary truncate">
                    {truncate(paper.filename, 28)}
                  </p>
                  <p className="text-xs text-text-tertiary">
                    {formatDate(paper.created_at)}
                  </p>
                </div>

                {confirmDeleteId === paper.id ? (
                  <div className="flex items-center gap-1 shrink-0" onClick={e => e.stopPropagation()}>
                    <button
                      onClick={() => handleDelete(paper.id)}
                      className="text-xs text-red-500 hover:text-red-600 px-1.5 py-0.5 rounded hover:bg-red-50 transition-colors"
                    >
                      Delete
                    </button>
                    <button
                      onClick={() => setConfirmDeleteId(null)}
                      className="text-xs text-text-tertiary hover:text-text-secondary px-1 py-0.5 rounded hover:bg-surface-tertiary transition-colors"
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={e => {
                      e.stopPropagation()
                      setConfirmDeleteId(paper.id)
                    }}
                    className="opacity-0 group-hover:opacity-100 text-text-tertiary hover:text-red-500 p-1 rounded hover:bg-surface-tertiary transition-all shrink-0"
                  >
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M18 6L6 18M6 6l12 12" />
                    </svg>
                  </button>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </>
  )
}
