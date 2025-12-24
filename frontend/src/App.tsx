import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import './App.css'
import type { ChatMessage, StatusResponse, ChatRequest, ChatResponse } from './types'

const API_BASE = 'http://localhost:8000'

type ProcessingStatus = 'queued' | 'processing' | 'completed' | 'error' | null

interface JobStatus {
  jobId: string
  filename: string
  status: ProcessingStatus
  progress?: string
  error?: string
}

function App() {
  const [files, setFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState<boolean>(false)
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
    // Carica le collections disponibili all'avvio
    fetchCollections()
  }, [])

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
    setFiles(selectedFiles)
    setJobStatuses([])
    setMessages([])
    setChatStarted(false)
  }

  const handleUpload = async (): Promise<void> => {
    if (!files.length) {
      alert('Seleziona almeno un file PDF')
      return
    }
  
    setUploading(true)
    const newJobStatuses: JobStatus[] = files.map(file => ({
      jobId: '',
      filename: file.name,
      status: 'queued' as ProcessingStatus,
      progress: 'In preparazione...'
    }))
    setJobStatuses(newJobStatuses)
  
    try {
      const formData = new FormData()
      
      // IMPORTANTE: Usa lo stesso nome che il backend si aspetta
      if (files.length === 1) {
        // Singolo file
        formData.append('file', files[0])
      } else {
        // Multipli file - ognuno con lo stesso nome 'file'
        files.forEach(file => {
          formData.append('file', file)
        })
      }
      
      // Aggiungi collection_name se supportato
      formData.append('collection_name', activeCollection)
      
      console.log('Invio FormData con', files.length, 'file(s)')
      
      // Prova diverse strategie:
      const response = await axios.post(`${API_BASE}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        // Aggiungi timeout
        timeout: 30000,
      })
  
      console.log('Risposta dal server:', response.data)
      
      // Gestisci la risposta in base al formato
      const responseData = response.data
      
      if (Array.isArray(responseData)) {
        // Risposta multipla
        setJobStatuses(prev => prev.map((job, index) => {
          const uploadResp = responseData[index]
          return {
            ...job,
            jobId: uploadResp.job_id || uploadResp.jobId,
            status: 'queued',
            progress: 'In coda...'
          }
        }))
        
        // Avvia polling per ogni job
        responseData.forEach((uploadResp, index) => {
          if (uploadResp.job_id || uploadResp.jobId) {
            pollStatus(uploadResp.job_id || uploadResp.jobId, index)
          }
        })
      } else {
        // Risposta singola
        setJobStatuses(prev => prev.map((job, index) => {
          if (index === 0) {
            return {
              ...job,
              jobId: responseData.job_id || responseData.jobId,
              status: 'queued',
              progress: 'In coda...'
            }
          }
          return job
        }))
        
        if (responseData.job_id || responseData.jobId) {
          pollStatus(responseData.job_id || responseData.jobId, 0)
        }
      }
      
    } catch (error) {
      console.error('Errore dettagliato upload:', error)
      
      if (axios.isAxiosError(error)) {
        console.error('Status:', error.response?.status)
        console.error('Data:', error.response?.data)
        console.error('Headers:', error.response?.headers)
        
        let errorMessage = 'Errore durante il caricamento'
        
        if (error.response?.status === 422) {
          errorMessage = 'Errore di validazione: ' + JSON.stringify(error.response.data)
        } else if (error.response?.data?.detail) {
          errorMessage = error.response.data.detail
        } else {
          errorMessage = error.message
        }
        
        alert(errorMessage)
      } else {
        alert('Errore sconosciuto durante il caricamento')
      }
      
      setUploading(false)
    }
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
                status: newStatus.status as ProcessingStatus, 
                progress: newStatus.progress,
                error: newStatus.error
              }
            : job
        ))
        
        if (newStatus.status === 'completed' || newStatus.status === 'error') {
          clearInterval(interval)
          
          // Controlla se tutti i job sono completati
          setTimeout(() => {
            setJobStatuses(prev => {
              const allJobs = [...prev]
              const allCompleted = allJobs.every(job => 
                job.status === 'completed' || job.status === 'error'
              )
              
              if (allCompleted && !chatStarted) {
                setUploading(false)
                setMessages([{
                  role: 'assistant',
                  content: `✅ ${getCompletionCount()} documento(i) processati con successo! Ora puoi farmi domande sui bandi.`
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
    setJobStatuses([])
  }

  const handleRemoveFile = (index: number): void => {
    setFiles(prev => prev.filter((_, i) => i !== index))
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

  return (
    <div className="app">
      <div className="container">
        <header className="header">
          <div className="header-content">
            <h1>📄 GraphRAG Bandi Chat</h1>
            <p className="subtitle">Carica multipli PDF e chatta con l'AI per ottenere informazioni specifiche</p>
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
          {/* Upload Section */}
          <section className="upload-card card">
            <div className="card-header">
              <h2>📤 Carica Documenti</h2>
              <span className="card-subtitle">(Puoi caricare uno o più PDF contemporaneamente)</span>
            </div>
            <div className="upload-section">
              <div className="file-input-container multiple">
                <label className="file-input-label multiple">
                  <input
                    type="file"
                    accept=".pdf"
                    onChange={handleFileChange}
                    disabled={uploading}
                    className="file-input"
                    multiple
                  />
                  <span className="file-input-custom">
                    {files.length > 0 
                      ? `${files.length} file selezionati` 
                      : 'Scegli file PDF (multipli)...'}
                  </span>
                </label>
                
                {files.length > 0 && (
                  <div className="selected-files">
                    <h4>File selezionati ({files.length}):</h4>
                    <ul className="file-list">
                      {files.map((file, index) => (
                        <li key={index} className="file-item">
                          <span className="file-name">{file.name}</span>
                          <span className="file-size">
                            {(file.size / 1024 / 1024).toFixed(2)} MB
                          </span>
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
                    ) : `Carica ${files.length} PDF`}
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
                            <span className="job-name">{job.filename}</span>
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

          {/* Chat Section */}
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
                      <p>Sto caricando e processando {files.length} documento(i). Puoi comunque chattare con i documenti già processati.</p>
                    </div>
                  </div>
                </div>
              )}
              
              {!uploading && messages.length === 0 && jobStatuses.length === 0 && (
                <div className="welcome-notice">
                  <div className="notice-content">
                    <span className="notice-icon">💡</span>
                    <div>
                      <strong>Benvenuto in GraphRAG Bandi Chat!</strong>
                      <p>Carica uno o più documenti PDF per iniziare una nuova conversazione, oppure fai direttamente una domanda sui documenti già presenti nel sistema.</p>
                      <p className="hint-text">
                        💡 Puoi caricare più PDF contemporaneamente per confrontare bandi diversi!
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
                  <span>GraphRAG analizzerà tutti i documenti e risponderà alle tue domande</span>
                </div>
              </div>
            </div>
          </section>
        </main>

        <footer className="footer">
          <p>GraphRAG Bandi Chat v2.0 • Supporto multi-documento con AI avanzata</p>
          <div className="footer-info">
            <span>📚 {availableCollections.length} collections disponibili</span>
            <span>•</span>
            <span>⚡ Processing parallelo con Celery</span>
          </div>
        </footer>
      </div>
    </div>
  )
}

export default App