import { useEffect, useRef } from 'react'

const COLORS = [
  { name: 'yellow', hex: '#fef08a' },
  { name: 'green', hex: '#bbf7d0' },
  { name: 'blue', hex: '#bfdbfe' },
  { name: 'pink', hex: '#fbcfe8' },
  { name: 'orange', hex: '#fed7aa' },
]

interface AnnotationPopoverProps {
  x: number
  y: number
  onHighlight: (color: string, selectedText: string, pageNumber: number) => void
  selectedText: string
  pageNumber: number
  onDismiss: () => void
}

export default function AnnotationPopover({
  x,
  y,
  onHighlight,
  selectedText,
  pageNumber,
  onDismiss,
}: AnnotationPopoverProps) {
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        onDismiss()
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [onDismiss])

  return (
    <div
      ref={ref}
      className="fixed z-50 bg-surface border border-border rounded-lg shadow-lg px-2 py-1.5 flex items-center gap-1.5"
      style={{ left: x, top: y }}
    >
      {COLORS.map(color => (
        <button
          key={color.name}
          onClick={() => onHighlight(color.hex, selectedText, pageNumber)}
          className="w-6 h-6 rounded-full border border-border hover:scale-110 transition-transform"
          style={{ backgroundColor: color.hex }}
          title={color.name}
        />
      ))}
    </div>
  )
}
