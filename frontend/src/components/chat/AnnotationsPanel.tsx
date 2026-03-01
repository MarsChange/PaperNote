import type { Annotation } from '../../types'

interface AnnotationsPanelProps {
  annotations: Annotation[]
  onDelete: (annotationId: string) => void
  onGoToPage: (pageNumber: number) => void
}

export default function AnnotationsPanel({ annotations, onDelete, onGoToPage }: AnnotationsPanelProps) {
  const truncate = (str: string, max: number) =>
    str.length > max ? str.slice(0, max) + '...' : str

  if (annotations.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center px-4">
        <p className="text-sm text-text-tertiary">No annotations yet</p>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-3 space-y-2 min-h-0">
      {annotations.map(ann => (
        <div
          key={ann.id}
          className="group flex items-start gap-2 p-2 rounded-lg hover:bg-surface-secondary cursor-pointer transition-colors"
          onClick={() => onGoToPage(ann.page_number)}
        >
          <span
            className="w-2.5 h-2.5 rounded-full shrink-0 mt-1"
            style={{ backgroundColor: ann.color }}
          />
          <div className="flex-1 min-w-0">
            <p className="text-sm text-text-primary leading-snug">
              {truncate(ann.text_content, 60)}
            </p>
          </div>
          <span className="text-xs text-text-tertiary bg-surface-tertiary px-1.5 py-0.5 rounded shrink-0">
            p.{ann.page_number}
          </span>
          <button
            onClick={e => {
              e.stopPropagation()
              onDelete(ann.id)
            }}
            className="opacity-0 group-hover:opacity-100 text-text-tertiary hover:text-red-500 p-0.5 rounded transition-all shrink-0"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
      ))}
    </div>
  )
}
