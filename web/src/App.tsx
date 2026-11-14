// strata-models: one question per area. Overview: is the GPU free, and who is using it? Models: which model serves each role, and is it loaded?
// Jobs: what ran, for whom, how long, and why did it fail? Evals: which model, settings or prompt should serve a role, with what evidence? Usage: which apps use the GPU most? Cloud: what the API keys spend, and on what? Settings: where weights live and how much VRAM to keep free.
import { Navigate, Route, Routes } from 'react-router'
import { AppShell, useLoad } from 'strata-react'
import { Activity, Boxes, Cloud, Cpu, FlaskConical, History, ImagePlus, Settings, Gauge } from 'lucide-react'
import { api } from './api'
import OverviewPage from './components/OverviewPage'
import ModelsPage from './components/ModelsPage'
import JobsPage from './components/JobsPage'
import UsagePage from './components/UsagePage'
import CloudPage from './components/CloudPage'
import SettingsPage from './components/SettingsPage'
import EvalsPage from './components/EvalsPage'
import GeneratePage from './components/GeneratePage'
import EvalPage from './components/EvalPage'

const nav = [
  { to: '/', label: 'Overview', icon: Gauge, end: true },
  { to: '/generate', label: 'Generate', icon: ImagePlus },
  { to: '/models', label: 'Models', icon: Boxes },
  { to: '/jobs', label: 'Jobs', icon: History },
  { to: '/evals', label: 'Evals', icon: FlaskConical },
  { to: '/usage', label: 'Usage', icon: Activity },
  { to: '/cloud', label: 'Cloud', icon: Cloud },
  { to: '/settings', label: 'Settings', icon: Settings },
]

export default function App() {
  const health = useLoad({ queryKey: ['health'], queryFn: api.health, staleTime: Infinity })
  return (
    <AppShell product={<span className="inline-flex items-center gap-2"><Cpu aria-hidden className="size-5" />Strata Models</span>} nav={nav} topBar={health.status === 'ready' ? <span className="type-caption text-muted-foreground font-mono">v{health.data.version}</span> : undefined}>
      <Routes>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/generate" element={<GeneratePage />} />
        <Route path="/models" element={<ModelsPage />} />
        <Route path="/jobs" element={<JobsPage />} />
        <Route path="/evals" element={<EvalsPage />} />
        <Route path="/evals/:name" element={<EvalPage />} />
        <Route path="/usage" element={<UsagePage />} />
        <Route path="/cloud" element={<CloudPage />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AppShell>
  )
}
