import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter } from 'react-router'
import { StrataQueryProvider, ThemeProvider } from 'strata-react'
import './index.css'
import App from './App'

createRoot(document.getElementById('root')!).render(
  <StrictMode><ThemeProvider><StrataQueryProvider><HashRouter><App /></HashRouter></StrataQueryProvider></ThemeProvider></StrictMode>,
)
