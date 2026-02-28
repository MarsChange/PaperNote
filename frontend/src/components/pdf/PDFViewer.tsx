import { useState } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'

pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`

interface PDFViewerProps {
  url: string
}

export default function PDFViewer({ url }: PDFViewerProps) {
  const [numPages, setNumPages] = useState<number>(0)
  const [scale, setScale] = useState(1.2)

  return (
    <div className="h-full flex flex-col bg-surface-secondary">
      {/* Toolbar */}
      <div className="h-10 flex items-center justify-center gap-3 border-b border-border-light bg-surface px-3 shrink-0">
        <button
          onClick={() => setScale(s => Math.max(0.5, s - 0.1))}
          className="text-text-secondary hover:text-text-primary p-1 rounded hover:bg-surface-tertiary transition-colors text-sm"
        >
          −
        </button>
        <span className="text-xs text-text-secondary w-12 text-center">
          {Math.round(scale * 100)}%
        </span>
        <button
          onClick={() => setScale(s => Math.min(3, s + 0.1))}
          className="text-text-secondary hover:text-text-primary p-1 rounded hover:bg-surface-tertiary transition-colors text-sm"
        >
          +
        </button>
        <span className="text-xs text-text-tertiary mx-2">|</span>
        <span className="text-xs text-text-tertiary">
          {numPages > 0 ? `${numPages} pages` : ''}
        </span>
      </div>

      {/* PDF content */}
      <div className="flex-1 overflow-auto flex justify-center py-4">
        <Document
          file={url}
          onLoadSuccess={({ numPages: n }) => setNumPages(n)}
          loading={
            <div className="flex items-center justify-center h-full">
              <p className="text-sm text-text-tertiary">Loading PDF...</p>
            </div>
          }
          error={
            <div className="flex items-center justify-center h-full">
              <p className="text-sm text-red-500">Failed to load PDF</p>
            </div>
          }
        >
          <div className="flex flex-col items-center gap-4">
            {Array.from({ length: numPages }, (_, i) => (
              <Page
                key={i + 1}
                pageNumber={i + 1}
                scale={scale}
                className="shadow-sm"
                renderTextLayer={true}
                renderAnnotationLayer={true}
              />
            ))}
          </div>
        </Document>
      </div>
    </div>
  )
}