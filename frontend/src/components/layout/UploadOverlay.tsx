import { useCallback, useState } from 'react'

interface UploadOverlayProps {
  onFileSelect: (file: File) => void
  onOpenSettings: () => void
}

export default function UploadOverlay({ onFileSelect, onOpenSettings }: UploadOverlayProps) {
  const [isDragging, setIsDragging] = useState(false)

  const handleFile = useCallback((file: File) => {
    if (file.type === 'application/pdf') {
      onFileSelect(file)
    }
  }, [onFileSelect])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  return (
    <div
      className="h-full w-full flex flex-col items-center justify-center bg-surface"
      onDragOver={e => { e.preventDefault(); setIsDragging(true) }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
    >
      {/* Settings button */}
      <button
        onClick={onOpenSettings}
        className="absolute top-4 right-4 text-text-tertiary hover:text-text-secondary p-2 rounded-lg hover:bg-surface-tertiary transition-colors"
        title="Settings"
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="3"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/>
        </svg>
      </button>

      <div className="flex flex-col items-center gap-6 max-w-md px-8">
        {/* Logo */}
        <div className="flex flex-col items-center gap-1">
          <h1 className="text-xl font-semibold text-text-primary tracking-tight">PaperNote</h1>
          <p className="text-sm text-text-tertiary">AI-powered paper reading assistant</p>
        </div>

        {/* Upload area */}
        <label
          className={`
            w-full aspect-[2/1] flex flex-col items-center justify-center gap-3
            rounded-xl border-2 border-dashed cursor-pointer transition-all
            ${isDragging
              ? 'border-accent bg-accent-light scale-[1.02]'
              : 'border-border hover:border-text-tertiary hover:bg-surface-secondary'
            }
          `}
        >
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-text-tertiary">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="12" y1="18" x2="12" y2="12" />
            <line x1="9" y1="15" x2="12" y2="12" />
            <line x1="15" y1="15" x2="12" y2="12" />
          </svg>
          <div className="text-center">
            <p className="text-sm text-text-secondary">
              {isDragging ? 'Drop PDF here' : 'Drop PDF here or click to browse'}
            </p>
            <p className="text-xs text-text-tertiary mt-1">Supports .pdf files</p>
          </div>
          <input
            type="file"
            accept=".pdf"
            className="hidden"
            onChange={e => {
              const file = e.target.files?.[0]
              if (file) handleFile(file)
            }}
          />
        </label>
      </div>
    </div>
  )
}