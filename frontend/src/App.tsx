import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import './App.css'
import type { 
  ChatMessage, 
  StatusResponse, 
  ChatRequest, 
  ChatResponse,
  UploadResponse,
  UploadInfo 
} from './types'

const API_BASE = 'http://localhost:8000'

type ProcessingStatus = 'queued' | 'processing' | 'completed' | 'error' | null

interface JobStatus {
  jobId: string
  filename: string
  originalPath?: string
  status: ProcessingStatus
  progress?: string
  error?: string
}

function App() {
  const [files, setFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState<boolean>(false)
  const [uploadInfo, setUploadInfo] = useState<UploadInfo | null>(null)
  const [jobStatuses, setJobStatuses] = useState<JobStatus[]>([])
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [inputMessage, setInputMessage] = useState<string>('')
  const [loadingAnswer, setLoadingAnswer] = useState<boolean>(false)
  const [chatStarted, setChatStarted] = useState<boolean>(false)
  const [activeCollection, setActiveCollection] = useState<string>('Bandi')
  const [availableCollections, setAvailableCollections] = useState<string[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = (): void => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    fetchCollections()
  }, [])

  useEffect(() => {
    if (files.length === 0) {
      setUploadInfo(null)
      return
    }

    const info: UploadInfo = {
      fileCount: files.length,
      totalSize: files.reduce((sum, file) => sum + file.size, 0),
      type: 'file',
      zipContentCount: 0
    }

    // Analizza il tipo di upload
    const hasRelativePath = files.some(f => f.webkitRelativePath && f.webkitRelativePath !== f.name)
    const allArePDF = files.every(f => f.name.toLowerCase().endsWith('.pdf'))
    const hasZIP = files.some(f => f.name.toLowerCase().endsWith('.zip'))
    const allAreZIP = files.length === 1 && hasZIP
    const hasNonPDF = files.some(f => !f.name.toLowerCase().endsWith('.pdf') && !f.name.toLowerCase().endsWith('.zip'))

    if (hasNonPDF) {
      // Se ci sono file non PDF e non ZIP, è un errore
      info.type = 'mixed'
      alert('⚠️ Attenzione: Solo file PDF e ZIP sono supportati. I file non supportati verranno ignorati.')
    } else if (allAreZIP) {
      info.type = 'zip'
    } else if (hasRelativePath && allArePDF) {
      info.type = 'folder'
    } else if (hasZIP && files.length > 1) {
      info.type = 'mixed'
    } else if (allArePDF) {
      info.type = 'file'
    } else {
      info.type = 'mixed'
    }

    setUploadInfo(info)
  }, [files])

  const fetchCollections = async (): Promise<void> => {
    try {
      const response = await axios.get(`${API_BASE}/collections`)
      setAvailableCollections(response.data.collections || [])
    } catch (error) {
      console.error('Errore nel caricamento delle collections:', error)
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    const selectedFiles = Array.from(e.target.files || [])
    
    // Filtra solo PDF e ZIP
    const validFiles = selectedFiles.filter(file => 
      file.name.toLowerCase().endsWith('.pdf') || 
      file.name.toLowerCase().endsWith('.zip')
    )
    
    // Se ci sono file non validi, mostra avviso
    if (validFiles.length < selectedFiles.length) {
      alert('⚠️ Solo file PDF e ZIP sono supportati. Gli altri file sono stati ignorati.')
    }
    
    setFiles(validFiles)
    setJobStatuses([])
    setMessages([])
    setChatStarted(false)
  }

  const handleUpload = async (): Promise<void> => {
    if (!files.length) {
      alert('Seleziona almeno un file PDF o ZIP')
      return
    }

    // Verifica che tutti i file siano PDF o ZIP
    const hasInvalidFiles = files.some(file => 
      !file.name.toLowerCase().endsWith('.pdf') && 
      !file.name.toLowerCase().endsWith('.zip')
    )

    if (hasInvalidFiles) {
      alert('Errore: Solo file PDF e ZIP sono supportati')
      return
    }

    setUploading(true)
    
    // Filtra solo i PDF per lo stato (gli ZIP verranno estratti dal backend)
    const pdfFiles = files.filter(f => f.name.toLowerCase().endsWith('.pdf'))
    const initialJobCount = uploadInfo?.type === 'zip' ? 1 : pdfFiles.length
    
    const newJobStatuses: JobStatus[] = Array.from({ length: initialJobCount }, (_, index) => ({
      jobId: '',
      filename: uploadInfo?.type === 'zip' 
        ? files[0].name 
        : (pdfFiles[index]?.name || ''),
      originalPath: pdfFiles[index]?.webkitRelativePath || undefined,
      status: 'queued' as ProcessingStatus,
      progress: 'In preparazione...'
    }))
    
    setJobStatuses(newJobStatuses)

    try {
      const formData = new FormData()
      const info = uploadInfo || analyzeFiles(files)

      // Logica di invio per adattarsi al backend
      if (info.type === 'zip') {
        // Singolo ZIP - usa 'file' come parametro
        formData.append('file', files[0])
      } else if (info.type === 'folder') {
        // Cartella - invia tutti i file ma con nome 'file' (non 'files')
        files.forEach(file => {
          formData.append('file', file)
        })
      } else {
        // File singoli o multipli
        files.forEach(file => {
          formData.append('file', file)
        })
      }

      // Aggiungi collection
      formData.append('collection_name', activeCollection)

      console.log(`Invio ${files.length} file(s) - Tipo: ${info.type}`)

      const response = await axios.post<UploadResponse | UploadResponse[]>(`${API_BASE}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 60000,
      })

      console.log('Risposta dal server:', response.data)

      const responseData = response.data
      const isArray = Array.isArray(responseData)
      const jobs = isArray ? responseData : [responseData]

      // Aggiorna job statuses con la risposta reale dal backend
      setJobStatuses(prev => jobs.map((uploadResp, index) => {
        const existingJob = prev[index] || prev[0] || {}
        return {
          jobId: uploadResp.job_id || '',
          filename: existingJob.filename || uploadResp.message.split(': ')[1] || `Documento ${index + 1}`,
          originalPath: existingJob.originalPath,
          status: 'queued' as ProcessingStatus,
          progress: info.type === 'zip' ? 'Estrazione in corso...' : 'In coda...'
        }
      }))

      // Avvia polling per ogni job
      jobs.forEach((uploadResp, index) => {
        if (uploadResp.job_id) {
          pollStatus(uploadResp.job_id, index)
        }
      })

    } catch (error) {
      console.error('Errore upload:', error)
      
      if (axios.isAxiosError(error)) {
        let errorMessage = 'Errore durante il caricamento'
        
        if (error.response?.status === 400) {
          errorMessage = error.response.data.detail || 'Formato file non supportato'
        } else if (error.response?.status === 422) {
          errorMessage = 'Errore di validazione nei dati inviati'
        } else if (error.response?.data?.detail) {
          errorMessage = error.response.data.detail
        } else {
          errorMessage = error.message
        }
        
        alert(`❌ ${errorMessage}`)
      } else {
        alert('❌ Errore sconosciuto durante il caricamento')
      }
      
      setUploading(false)
      setJobStatuses([])
    }
  }

  const analyzeFiles = (files: File[]): UploadInfo => {
    const info: UploadInfo = {
      fileCount: files.length,
      totalSize: files.reduce((sum, file) => sum + file.size, 0),
      type: 'file',
      zipContentCount: 0
    }

    const hasRelativePath = files.some(f => f.webkitRelativePath && f.webkitRelativePath !== f.name)
    const allArePDF = files.every(f => f.name.toLowerCase().endsWith('.pdf'))
    const hasZIP = files.some(f => f.name.toLowerCase().endsWith('.zip'))
    const allAreZIP = files.length === 1 && hasZIP

    if (allAreZIP) {
      info.type = 'zip'
    } else if (hasRelativePath && allArePDF) {
      info.type = 'folder'
    } else if (hasZIP && files.length > 1) {
      info.type = 'mixed'
    } else if (allArePDF) {
      info.type = 'file'
    } else {
      info.type = 'mixed'
    }

    return info
  }

  const pollStatus = async (jobId: string, index: number): Promise<void> => {
    const interval = setInterval(async () => {
      try {
        const response = await axios.get<StatusResponse>(`${API_BASE}/status/${jobId}`)
        const newStatus = response.data
        
        setJobStatuses(prev => prev.map((job, idx) => 
          idx === index 
            ? { 
                ...job, 
                jobId: newStatus.job_id,
                status: newStatus.status,
                progress: newStatus.progress,
                error: newStatus.error
              }
            : job
        ))
        
        if (newStatus.status === 'completed' || newStatus.status === 'error') {
          clearInterval(interval)
          
          setTimeout(() => {
            setJobStatuses(prev => {
              const allJobs = [...prev]
              const allCompleted = allJobs.every(job => 
                job.status === 'completed' || job.status === 'error'
              )
              
              if (allCompleted && !chatStarted) {
                setUploading(false)
                const successCount = allJobs.filter(job => job.status === 'completed').length
                setMessages([{
                  role: 'assistant',
                  content: getUploadSuccessMessage(successCount, uploadInfo)
                }])
                setChatStarted(true)
                fetchCollections()
              }
              return allJobs
            })
          }, 500)
        }
      } catch (error) {
        console.error('Errore polling status:', error)
        clearInterval(interval)
      }
    }, 2000)
  }

  const getUploadSuccessMessage = (successCount: number, info: UploadInfo | null): string => {
    if (!info) {
      return `✅ ${successCount} documento(i) processati con successo! Ora puoi farmi domande.`
    }

    switch (info.type) {
      case 'zip':
        return `✅ Archivio ZIP processato con successo! ${successCount} documento(i) estratti e pronti per le domande.`
      case 'folder':
        return `✅ Cartella processata con successo! ${successCount} documento(i) caricati e pronti per le domande.`
      case 'mixed':
        return `✅ ${successCount} file processati con successo! Pronto per le domande.`
      default:
        return `✅ ${successCount === 1 ? 'Documento' : 'Documenti'} processato(i) con successo! Ora puoi farmi domande.`
    }
  }

  const getUploadTypeIcon = (): string => {
    if (!uploadInfo) return '📄'
    
    switch (uploadInfo.type) {
      case 'zip': return '📦'
      case 'folder': return '📁'
      case 'mixed': return '📚'
      default: return '📄'
    }
  }

  const getUploadTypeText = (): string => {
    if (!uploadInfo) return 'file'
    
    switch (uploadInfo.type) {
      case 'zip': return 'archivio ZIP'
      case 'folder': return 'cartella'
      case 'mixed': return 'file misti'
      default: return 'PDF'
    }
  }

  const handleSendMessage = async (): Promise<void> => {
    if (!inputMessage.trim() || loadingAnswer) return

    const userMessage: ChatMessage = {
      role: 'user',
      content: inputMessage.trim()
    }

    setMessages(prev => [...prev, userMessage])
    setInputMessage('')
    setLoadingAnswer(true)

    try {
      const allMessages: ChatMessage[] = [...messages, userMessage]

      const request: ChatRequest = {
        messages: allMessages,
        collection_name: activeCollection
      }

      const response = await axios.post<ChatResponse>(`${API_BASE}/chat`, request)

      setMessages(prev => [...prev, response.data.message])
    } catch (error) {
      console.error('Errore chat:', error)
      const errorMessage = axios.isAxiosError(error)
        ? error.response?.data?.detail || error.message
        : 'Errore sconosciuto'
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '❌ Errore durante l\'elaborazione: ' + errorMessage
      }])
    } finally {
      setLoadingAnswer(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const startNewChat = (): void => {
    setMessages([])
    setChatStarted(false)
    setInputMessage('')
    setFiles([])
    setUploadInfo(null)
    setJobStatuses([])
  }

  const handleRemoveFile = (index: number): void => {
    const newFiles = [...files]
    newFiles.splice(index, 1)
    setFiles(newFiles)
    
    if (newFiles.length === 0) {
      setUploadInfo(null)
    }
  }

  const handleCollectionChange = (collection: string): void => {
    setActiveCollection(collection)
    setMessages([{
      role: 'assistant',
      content: `✅ Collection cambiata a "${collection}". Puoi ora chattare con i documenti in questa collection.`
    }])
  }

  const getCompletionCount = (): number => {
    return jobStatuses.filter(job => job.status === 'completed').length
  }

  const getTotalJobs = (): number => {
    return jobStatuses.length
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  // Modifica per risolvere l'errore TypeScript
  const fileInputProps: any = {
    type: "file",
    accept: ".pdf,.zip",
    onChange: handleFileChange,
    disabled: uploading,
    className: "file-input",
    multiple: true
  }

  // Aggiungi attributi non standard solo se supportati
  if (typeof document !== 'undefined') {
    const input = document.createElement('input')
    if ('webkitdirectory' in input) {
      fileInputProps.webkitdirectory = ""
      fileInputProps.directory = ""
    }
  }

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <div className="header-content">
            <h1>📄 GraphRAG Bandi Chat</h1>
            <p className="subtitle">Carica PDF o ZIP - il sistema capisce automaticamente</p>
          </div>
          <div className="header-actions">
            {availableCollections.length > 0 && (
              <div className="collection-selector">
                <select 
                  value={activeCollection}
                  onChange={(e) => handleCollectionChange(e.target.value)}
                  className="collection-select"
                >
                  {availableCollections.map(collection => (
                    <option key={collection} value={collection}>
                      {collection}
                    </option>
                  ))}
                </select>
                <span className="collection-label">Collection attiva</span>
              </div>
            )}
          </div>
        </header>

        <main className="main-content">
          <section className="upload-card card">
            <div className="card-header">
              <h2>📤 Carica Documenti</h2>
              <span className="card-subtitle">Supporta PDF singoli, multipli, cartelle e ZIP</span>
            </div>
            <div className="upload-section">
              <div className="file-input-container multiple">
                <label className="file-input-label multiple">
                  <input
                    {...fileInputProps}
                  />
                  <span className="file-input-custom">
                    {files.length > 0 
                      ? `${files.length} file selezionati` 
                      : 'Trascina qui o clicca per selezionare PDF o ZIP'}
                  </span>
                </label>
                
                {uploadInfo && (
                  <div className="upload-info-banner">
                    <div className="upload-info-content">
                      <span className="upload-icon">{getUploadTypeIcon()}</span>
                      <div className="upload-info-details">
                        <div className="upload-type">
                          <strong>Tipo rilevato:</strong> {getUploadTypeText()}
                        </div>
                        <div className="upload-stats">
                          <span>{uploadInfo.fileCount} file</span>
                          <span>•</span>
                          <span>{formatFileSize(uploadInfo.totalSize)}</span>
                        </div>
                      </div>
                    </div>
                    <div className="upload-hint">
                      💡 Supportato: PDF singoli • Multipli PDF • Cartelle (se browser supporta) • Archivi ZIP
                    </div>
                  </div>
                )}

                {files.length > 0 && (
                  <div className="selected-files">
                    <h4>File selezionati ({files.length}):</h4>
                    <ul className="file-list">
                      {files.map((file, index) => (
                        <li key={index} className="file-item">
                          <div className="file-info">
                            <div className="file-header">
                              <span className="file-name">{file.name}</span>
                              <span className="file-size">
                                {formatFileSize(file.size)}
                              </span>
                            </div>
                            {file.webkitRelativePath && file.webkitRelativePath !== file.name && (
                              <span className="file-path">
                                {file.webkitRelativePath}
                              </span>
                            )}
                            <div className="file-type">
                              {file.name.toLowerCase().endsWith('.zip') ? (
                                <span className="file-type-badge zip">ZIP</span>
                              ) : file.name.toLowerCase().endsWith('.pdf') ? (
                                <span className="file-type-badge pdf">PDF</span>
                              ) : (
                                <span className="file-type-badge other">Altro</span>
                              )}
                            </div>
                          </div>
                          <button
                            onClick={() => handleRemoveFile(index)}
                            className="remove-file-btn"
                            disabled={uploading}
                            aria-label="Rimuovi file"
                          >
                            ×
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="upload-buttons">
                  <button
                    onClick={handleUpload}
                    disabled={uploading || !files.length}
                    className="btn btn-primary"
                  >
                    {uploading ? (
                      <>
                        <span className="spinner"></span>
                        Caricamento in corso...
                      </>
                    ) : (
                      `Carica ${files.length} file`
                    )}
                  </button>
                  {chatStarted && (
                    <button
                      onClick={startNewChat}
                      className="btn btn-secondary"
                    >
                      Nuova Chat
                    </button>
                  )}
                </div>
              </div>

              {jobStatuses.length > 0 && (
                <div className="status-section">
                  <div className="status-header">
                    <h3>Stato Elaborazione</h3>
                    <div className="status-summary">
                      <span className="status-summary-text">
                        {getCompletionCount()} / {getTotalJobs()} completati
                      </span>
                      <div className="progress-bar">
                        <div 
                          className="progress-fill"
                          style={{ width: `${(getCompletionCount() / getTotalJobs()) * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                  
                  <div className="jobs-list">
                    {jobStatuses.map((job, index) => (
                      <div key={index} className="job-item">
                        <div className="job-header">
                          <div className="job-filename">
                            <span className="job-index">{index + 1}.</span>
                            <div className="job-name-container">
                              <span className="job-name">{job.filename}</span>
                              {job.originalPath && job.originalPath !== job.filename && (
                                <span className="job-original-path">
                                  {job.originalPath}
                                </span>
                              )}
                            </div>
                          </div>
                          <div className={`job-status-indicator status-${job.status || ''}`}>
                            {job.status === 'queued' && '⏳'}
                            {job.status === 'processing' && '⚙️'}
                            {job.status === 'completed' && '✅'}
                            {job.status === 'error' && '❌'}
                            <span className="job-status-text">
                              {job.status === 'queued' && 'In coda'}
                              {job.status === 'processing' && 'Elaborazione...'}
                              {job.status === 'completed' && 'Completato'}
                              {job.status === 'error' && 'Errore'}
                            </span>
                          </div>
                        </div>
                        
                        <div className="job-details">
                          {job.progress && (
                            <div className="job-progress">
                              <span>{job.progress}</span>
                            </div>
                          )}
                          {job.error && (
                            <div className="job-error">
                              <strong>Errore:</strong> {job.error}
                            </div>
                          )}
                          {job.jobId && (
                            <div className="job-id">
                              <small>ID: {job.jobId}</small>
                            </div>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </section>

          <section className="chat-card card">
            <div className="card-header">
              <h2>💬 Chat con i Documenti</h2>
              <div className="chat-stats">
                {activeCollection && (
                  <span className="collection-badge">
                    Collection: {activeCollection}
                  </span>
                )}
                {messages.length > 0 && (
                  <span className="message-count">{messages.length} messaggi</span>
                )}
              </div>
            </div>

            <div className="chat-container">
              {uploading && (
                <div className="processing-notice">
                  <div className="notice-content">
                    <span className="notice-icon">📤</span>
                    <div>
                      <strong>Upload in corso...</strong>
                      <p>Sto caricando {uploadInfo ? getUploadTypeText() : 'documenti'}. 
                         Puoi comunque chattare con i documenti già processati.</p>
                    </div>
                  </div>
                </div>
              )}
              
              {!uploading && messages.length === 0 && jobStatuses.length === 0 && (
                <div className="welcome-notice">
                  <div className="notice-content">
                    <span className="notice-icon">🚀</span>
                    <div>
                      <strong>Carica e Chatta in modo Intelligente</strong>
                      <p>Seleziona PDF o ZIP. Il sistema riconosce automaticamente cosa hai caricato.</p>
                      <p className="hint-text">
                        💡 Supporta: PDF singoli • Multipli PDF • Cartelle (se supportato dal browser) • Archivi ZIP
                      </p>
                    </div>
                  </div>
                </div>
              )}
              
              <div className="chat-messages">
                {messages.map((msg, idx) => (
                  <div key={idx} className={`message ${msg.role}`}>
                    <div className="message-header">
                      <div className="message-avatar">
                        {msg.role === 'user' ? '👤' : '🤖'}
                      </div>
                      <span className="message-role">
                        {msg.role === 'user' ? 'Tu' : 'Assistente'}
                      </span>
                    </div>
                    <div className="message-content">
                      {msg.content}
                    </div>
                    <div className="message-time">
                      {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </div>
                ))}
                {loadingAnswer && (
                  <div className="message assistant">
                    <div className="message-header">
                      <div className="message-avatar">🤖</div>
                      <span className="message-role">Assistente</span>
                    </div>
                    <div className="message-content">
                      <div className="typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>

              <div className="chat-input-container">
                <div className="input-wrapper">
                  <textarea
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyDown={handleKeyPress}
                    placeholder="Scrivi la tua domanda sui bandi... (Premi Invio per inviare, Shift+Invio per andare a capo)"
                    className="chat-input"
                    rows={2}
                    disabled={loadingAnswer}
                  />
                  <button
                    onClick={handleSendMessage}
                    disabled={loadingAnswer || !inputMessage.trim()}
                    className="send-button"
                    aria-label="Invia messaggio"
                  >
                    {loadingAnswer ? (
                      <span className="sending-icon">⏳</span>
                    ) : (
                      <span className="send-icon">📤</span>
                    )}
                  </button>
                </div>
                <div className="input-hint">
                  <span>GraphRAG analizzerà automaticamente tutti i documenti caricati</span>
                </div>
              </div>
            </div>
          </section>
        </main>

        <footer className="footer">
          <p>GraphRAG Bandi Chat v3.0 • Riconoscimento automatico di file, cartelle e ZIP</p>
          <div className="footer-info">
            <span>📚 {availableCollections.length} collections</span>
            <span>•</span>
            <span>⚡ Processing intelligente</span>
            <span>•</span>
            <span>🤖 Riconosce automaticamente il tipo di upload</span>
          </div>
        </footer>
      </div>
    </div>
  )
}

export default App