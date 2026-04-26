import { useEffect, useRef, useState } from 'react'

const COLORS = [
  { name: 'Amber', hex: '#f4c95d' },
  { name: 'Mint', hex: '#87d4c2' },
  { name: 'Sky', hex: '#90c2ff' },
  { name: 'Rose', hex: '#f4a5a5' },
]

interface AnnotationPopoverProps {
  x: number
  y: number
  onHighlight: (color: string, selectedText: string, pageNumber: number, note?: string) => void
  onTranslate?: (selectedText: string, pageNumber: number) => Promise<string>
  selectedText: string
  pageNumber: number
  onDismiss: () => void
}

export default function AnnotationPopover({
  x,
  y,
  onHighlight,
  onTranslate,
  selectedText,
  pageNumber,
  onDismiss,
}: AnnotationPopoverProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [selectedColor, setSelectedColor] = useState(COLORS[0].hex)
  const [note, setNote] = useState('')
  const [translation, setTranslation] = useState('')
  const [isTranslating, setIsTranslating] = useState(Boolean(onTranslate))
  const [translateError, setTranslateError] = useState('')

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        onDismiss()
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [onDismiss])

  useEffect(() => {
    if (!onTranslate) {
      setIsTranslating(false)
      return
    }

    let cancelled = false
    setIsTranslating(true)
    setTranslateError('')

    onTranslate(selectedText, pageNumber)
      .then((result) => {
        if (!cancelled) setTranslation(result)
      })
      .catch(() => {
        if (!cancelled) setTranslateError('翻译暂时不可用')
      })
      .finally(() => {
        if (!cancelled) setIsTranslating(false)
      })

    return () => {
      cancelled = true
    }
  }, [onTranslate, pageNumber, selectedText])

  return (
    <div
      ref={ref}
      className="fixed z-50 w-[320px] rounded-[22px] border border-border bg-surface p-4 shadow-[0_20px_60px_rgba(36,30,20,0.18)] backdrop-blur"
      style={{ left: x, top: y }}
    >
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-semibold uppercase tracking-[0.24em] text-text-tertiary">
          Selection
        </span>
        <span className="rounded-full bg-surface-secondary px-2 py-1 text-[11px] text-text-secondary">
          p.{pageNumber}
        </span>
      </div>

      <p
        className="mt-3 text-sm leading-6 text-text-primary"
        style={{
          display: '-webkit-box',
          WebkitLineClamp: 3,
          WebkitBoxOrient: 'vertical',
          overflow: 'hidden',
        }}
      >
        {selectedText}
      </p>

      <div className="mt-3 rounded-2xl bg-surface-secondary px-3 py-2.5">
        <div className="mb-1 text-[11px] font-medium uppercase tracking-[0.18em] text-text-tertiary">
          Auto Translate
        </div>
        {isTranslating ? (
          <div className="flex items-center gap-2 text-sm text-text-secondary">
            <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-accent border-t-transparent" />
            正在翻译选中文本
          </div>
        ) : translateError ? (
          <p className="text-sm text-[#b1462f]">{translateError}</p>
        ) : (
          <p className="text-sm leading-6 text-text-secondary">
            {translation || '未获取到译文'}
          </p>
        )}
      </div>

      <div className="mt-3 flex items-center gap-2">
        {COLORS.map((color) => (
          <button
            key={color.name}
            type="button"
            onClick={() => setSelectedColor(color.hex)}
            className={`h-7 w-7 rounded-full border transition-transform ${
              selectedColor === color.hex
                ? 'scale-110 border-text-primary'
                : 'border-border hover:scale-105'
            }`}
            style={{ backgroundColor: color.hex }}
            title={color.name}
          />
        ))}

        <button
          type="button"
          onClick={() => onHighlight(selectedColor, selectedText, pageNumber)}
          className="ml-auto rounded-full bg-text-primary px-3 py-1.5 text-xs font-medium text-surface transition-colors hover:bg-[#30271e]"
        >
          仅高亮
        </button>
      </div>

      <textarea
        value={note}
        onChange={(e) => setNote(e.target.value)}
        rows={3}
        placeholder="写下你的理解、疑问或待办..."
        className="mt-3 w-full resize-none rounded-2xl border border-border bg-[#fffdfa] px-3 py-2.5 text-sm leading-6 text-text-primary outline-none transition-colors focus:border-accent"
      />

      <div className="mt-3 flex items-center justify-between">
        <button
          type="button"
          onClick={onDismiss}
          className="text-sm text-text-tertiary transition-colors hover:text-text-secondary"
        >
          关闭
        </button>
        <button
          type="button"
          onClick={() => onHighlight(selectedColor, selectedText, pageNumber, note.trim() || undefined)}
          className="rounded-full bg-accent px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-hover"
        >
          保存笔记
        </button>
      </div>
    </div>
  )
}
