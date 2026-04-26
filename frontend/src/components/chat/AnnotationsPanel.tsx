import type { Annotation } from '../../types'

interface AnnotationsPanelProps {
  annotations: Annotation[]
  onDelete: (annotationId: string) => void
  onGoToPage: (pageNumber: number) => void
}

export default function AnnotationsPanel({ annotations, onDelete, onGoToPage }: AnnotationsPanelProps) {
  if (annotations.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center px-6">
        <div className="max-w-xs text-center">
          <p className="text-base font-medium text-text-primary">还没有阅读笔记</p>
          <p className="mt-2 text-sm leading-6 text-text-tertiary">
            在 PDF 中选中文本后，可以直接高亮、翻译并记录你的理解。
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4">
      <div className="space-y-3">
        {annotations.map((annotation) => (
          <div
            key={annotation.id}
            onClick={() => onGoToPage(annotation.page_number)}
            className="group w-full cursor-pointer rounded-[22px] border border-border bg-surface-secondary p-4 text-left transition-all hover:border-text-tertiary hover:bg-surface-tertiary"
          >
            <div className="flex items-start gap-3">
              <span
                className="mt-1 h-3 w-3 shrink-0 rounded-full border border-white/80 shadow-sm"
                style={{ backgroundColor: annotation.color }}
              />

              <div className="min-w-0 flex-1">
                <div className="flex items-center justify-between gap-3">
                  <span className="rounded-full bg-surface px-2.5 py-1 text-[11px] font-medium uppercase tracking-[0.16em] text-text-tertiary">
                    Page {annotation.page_number}
                  </span>
                  <span className="text-xs text-text-tertiary">
                    {annotation.created_at ? new Date(annotation.created_at).toLocaleString() : ''}
                  </span>
                </div>

                <p className="mt-3 text-sm leading-6 text-text-primary">
                  {annotation.text_content}
                </p>

                {annotation.note && (
                  <div className="mt-3 rounded-2xl bg-surface px-3 py-2.5">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-text-tertiary">
                      Note
                    </p>
                    <p className="mt-1 text-sm leading-6 text-text-secondary">
                      {annotation.note}
                    </p>
                  </div>
                )}
              </div>

              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation()
                  onDelete(annotation.id)
                }}
                className="shrink-0 rounded-full p-2 text-text-tertiary opacity-0 transition-all hover:bg-surface hover:text-[#b1462f] group-hover:opacity-100"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M18 6L6 18M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
