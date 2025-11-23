export interface ChatMessage {
    role: 'user' | 'assistant'
    content: string
  }
  
  export interface UploadResponse {
    job_id: string
    message: string
    status: string
  }
  
  export interface StatusResponse {
    job_id: string
    status: 'queued' | 'processing' | 'completed' | 'error'
    progress?: string
    error?: string
    chunks_count?: number
    nodes_count?: number
    relationships_count?: number
    collection_name?: string
  }
  
  export interface ChatRequest {
    messages: ChatMessage[]
    collection_name?: string
  }
  
  export interface ChatResponse {
    message: ChatMessage
    graph_context?: {
      nodes: string[]
      edges: string[]
    }
  }
  
  export interface QueryRequest {
    query: string
    collection_name?: string
  }
  
  export interface QueryResponse {
    answer: string
    graph_context?: {
      nodes: string[]
      edges: string[]
    }
  }