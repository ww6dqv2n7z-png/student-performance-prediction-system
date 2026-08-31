'use client';
/* eslint-disable @next/next/no-img-element -- protected research charts are served by the authenticated local API */

import { FormEvent, useCallback, useEffect, useState } from 'react';

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

type View = 'dashboard' | 'students' | 'predict' | 'interventions' | 'model' | 'users' | 'settings';
type Theme = 'green' | 'blue' | 'slate' | 'maroon';
type User = { id: number; email: string; full_name: string; role: 'admin' | 'teacher' };
type Features = {
  gender: 'female' | 'male' | 'other'; age: number; attendance: number; study_time: number;
  previous_grade: number; internet_access: boolean; family_support: boolean; absences: number;
  participation: number; homework_completion: number;
};
type Student = Features & {
  id: number; student_code: string; department: string; academic_year: number; semester: number;
  academic_label?: string;
  risk_level?: 'low' | 'medium' | 'high'; pass_probability?: number; result_label?: string;
};
type Explanation = { feature: string; label: string; impact: number; direction: string };
type Prediction = {
  id: number; student_id?: number; student_code?: string; department?: string; result_label?: string;
  pass_probability?: number; predicted_final_grade?: number; risk_level: 'low' | 'medium' | 'high';
  explanation?: Explanation[]; created_at: string; created_by_name?: string;
};
type Intervention = {
  id: number; student_id: number; student_code: string; department: string; action: string; notes: string;
  status: 'pending' | 'in_progress' | 'completed'; assigned_to: string; due_date?: string; updated_at: string;
};
type Dashboard = {
  summary: { students: number; predictions: number; active_interventions: number; high_risk: number };
  risk_distribution: Record<string, number>;
  departments: { department: string; students: number; high_risk: number; avg_pass_probability?: number }[];
  cohorts: { academic_year: number; semester: number; label: string; students: number; high_risk: number; avg_pass_probability?: number }[];
  recent_predictions: Prediction[];
  model_metrics: Record<string, number | number[][]>;
  model_metadata: { dataset_label?: string; synthetic_data?: boolean; training_rows?: number; validation_rows?: number; test_rows?: number };
  research: { dataset_name?: string; source_rows?: number; best_model_by_f1?: string; models?: Record<string, Record<string, number | number[][]>>; fairness_gaps?: Record<string, Record<string, number>>; mtu_validation_status?: string };
  institution: { institution: string; department: string; department_code: string; mtu_validation_status: string };
};

const blankFeatures: Features = {
  gender: 'female', age: 18, attendance: 85, study_time: 6, previous_grade: 65,
  internet_access: true, family_support: true, absences: 5, participation: 70, homework_completion: 75,
};
const CEIT = 'Computer Engineering and Information Technology';
const academicLevels = [{year:1,semester:1,label:'First Year'},{year:2,semester:1,label:'Second Year'},{year:3,semester:1,label:'Third Year'},{year:4,semester:1,label:'Fourth Year'},{year:5,semester:1,label:'Fifth Year · First Semester'},{year:5,semester:2,label:'Fifth Year · Second Semester'},{year:6,semester:1,label:'Final Year'}];
const themeOptions: { id: Theme; label: string; color: string }[] = [{ id:'green', label:'Green', color:'#197354' },{ id:'blue', label:'Blue', color:'#315f98' },{ id:'slate', label:'Slate', color:'#4f6471' },{ id:'maroon', label:'Maroon', color:'#8b3d4d' }];
const blankStudent = { ...blankFeatures, student_code: '', department: CEIT, academic_year: 1, semester: 1 };

async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    ...options, credentials: 'include',
    headers: options.body ? { 'Content-Type': 'application/json', ...options.headers } : options.headers,
  });
  if (response.status === 401) throw new Error('AUTH_REQUIRED');
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail ?? 'Request failed');
  return payload as T;
}

function RiskBadge({ risk }: { risk?: string }) {
  if (!risk) return <span className="badge neutral">Not assessed</span>;
  return <span className={`badge ${risk}`}>{risk} risk</span>;
}

function Metric({ label, value, note, tone = '' }: { label: string; value: string | number; note?: string; tone?: string }) {
  return <article className={`metric-card ${tone}`}><span>{label}</span><strong>{value}</strong>{note && <small>{note}</small>}</article>;
}

function Empty({ title, text }: { title: string; text: string }) {
  return <div className="empty"><span>○</span><h3>{title}</h3><p>{text}</p></div>;
}

function FeatureFields({ value, onChange }: { value: Features; onChange: (next: Features) => void }) {
  const number = (name: keyof Features, next: number) => onChange({ ...value, [name]: next });
  return <div className="feature-grid">
    <label><span>Gender</span><select value={value.gender} onChange={(e) => onChange({ ...value, gender: e.target.value as Features['gender'] })}><option value="female">Female</option><option value="male">Male</option><option value="other">Other</option></select></label>
    <label><span>Age</span><input type="number" min="10" max="100" value={value.age} onChange={(e) => number('age', +e.target.value)} required /></label>
    <label><span>Attendance (%)</span><input type="number" min="0" max="100" step="0.1" value={value.attendance} onChange={(e) => number('attendance', +e.target.value)} required /></label>
    <label><span>Study time (hrs/week)</span><input type="number" min="0" max="168" step="0.5" value={value.study_time} onChange={(e) => number('study_time', +e.target.value)} required /></label>
    <label><span>Previous grade</span><input type="number" min="0" max="100" step="0.1" value={value.previous_grade} onChange={(e) => number('previous_grade', +e.target.value)} required /></label>
    <label><span>Absences</span><input type="number" min="0" max="365" value={value.absences} onChange={(e) => number('absences', +e.target.value)} required /></label>
    <label><span>Participation (%)</span><input type="number" min="0" max="100" step="0.1" value={value.participation} onChange={(e) => number('participation', +e.target.value)} required /></label>
    <label><span>Homework completion (%)</span><input type="number" min="0" max="100" step="0.1" value={value.homework_completion} onChange={(e) => number('homework_completion', +e.target.value)} required /></label>
    <label><span>Internet access</span><select value={String(value.internet_access)} onChange={(e) => onChange({ ...value, internet_access: e.target.value === 'true' })}><option value="true">Yes</option><option value="false">No</option></select></label>
    <label><span>Family support</span><select value={String(value.family_support)} onChange={(e) => onChange({ ...value, family_support: e.target.value === 'true' })}><option value="true">Yes</option><option value="false">No</option></select></label>
  </div>;
}

function Login({ setupNeeded, onLogin }: { setupNeeded: boolean; onLogin: (user: User) => void }) {
  const [email, setEmail] = useState(''); const [password, setPassword] = useState('');
  const [error, setError] = useState(''); const [loading, setLoading] = useState(false);
  async function submit(event: FormEvent) {
    event.preventDefault(); setLoading(true); setError('');
    try { const data = await api<{ user: User }>('/api/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) }); onLogin(data.user); }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Sign in failed'); }
    finally { setLoading(false); }
  }
  return <main className="login-page">
    <section className="login-brand"><div className="university-mark">CEIT</div><p>Mandalay Technological University</p><h1>Student Support<br />System</h1></section>
    <section className="login-panel"><form onSubmit={submit}>
      <p className="kicker">CEIT staff</p><h2>Sign in</h2>
      {setupNeeded && <div className="alert warning"><strong>Initial setup required</strong> Create the first administrator with the command in the README.</div>}
      {error && <div className="alert error">{error}</div>}
      <label><span>Email address</span><input type="email" autoComplete="username" value={email} onChange={(e) => setEmail(e.target.value)} required /></label>
      <label><span>Password</span><input type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} required /></label>
      <button className="primary full" disabled={loading}>{loading ? 'Signing in…' : 'Sign in securely'} <b>→</b></button>
      <small className="privacy">Local-only system · Session expires after 8 hours · Activity is audited</small>
    </form></section>
  </main>;
}

export default function Home() {
  const [user, setUser] = useState<User | null>(null); const [checking, setChecking] = useState(true);
  const [setupNeeded, setSetupNeeded] = useState(false); const [view, setView] = useState<View>('dashboard');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [theme, setTheme] = useState<Theme>('green');
  const [dashboard, setDashboard] = useState<Dashboard | null>(null); const [students, setStudents] = useState<Student[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]); const [interventions, setInterventions] = useState<Intervention[]>([]);
  const [notice, setNotice] = useState(''); const [error, setError] = useState(''); const [busy, setBusy] = useState(false);

  const signOut = useCallback(async () => { await api('/api/auth/logout', { method: 'POST' }).catch(() => null); setUser(null); }, []);
  const handleError = useCallback((reason: unknown) => {
    if (reason instanceof Error && reason.message === 'AUTH_REQUIRED') { setUser(null); return; }
    setError(reason instanceof Error ? reason.message : 'Something went wrong');
  }, []);
  const refresh = useCallback(async () => {
    if (!user) return;
    try {
      const [dash, studentData, predictionData, interventionData] = await Promise.all([
        api<Dashboard>('/api/dashboard'), api<{ students: Student[] }>('/api/students'),
        api<{ predictions: Prediction[] }>('/api/predictions'), api<{ interventions: Intervention[] }>('/api/interventions'),
      ]);
      setDashboard(dash); setStudents(studentData.students); setPredictions(predictionData.predictions); setInterventions(interventionData.interventions);
    } catch (reason) { handleError(reason); }
  }, [user, handleError]);

  useEffect(() => {
    Promise.all([fetch(`${API_URL}/api/health`).then((r) => r.json()), api<{ user: User }>('/api/auth/me')])
      .then(([health, auth]) => { setSetupNeeded(!health.admin_ready); setUser(auth.user); })
      .catch(async () => { try { const health = await fetch(`${API_URL}/api/health`).then((r) => r.json()); setSetupNeeded(!health.admin_ready); } catch { setSetupNeeded(true); } })
      .finally(() => setChecking(false));
  }, []);
  useEffect(() => {
    const timer = window.setTimeout(() => {
      const stored = window.localStorage.getItem('ceit-theme');
      if (themeOptions.some((option) => option.id === stored)) setTheme(stored as Theme);
    }, 0);
    return () => window.clearTimeout(timer);
  }, []);
  useEffect(() => {
    const timer = window.setTimeout(() => { void refresh(); }, 0);
    return () => window.clearTimeout(timer);
  }, [refresh]);

  if (checking) return <main className="loading-page"><div className="pulse" /><p>Opening MTU Student Support…</p></main>;
  if (!user) return <Login setupNeeded={setupNeeded} onLogin={setUser} />;

  const nav: { id: View; label: string; icon: string }[] = [
    { id: 'dashboard', label: 'Overview', icon: '⌂' }, { id: 'students', label: 'Students', icon: '◉' },
    { id: 'predict', label: 'Prediction', icon: '↗' }, { id: 'interventions', label: 'Interventions', icon: '✓' },
    { id: 'model', label: 'Model & reports', icon: '▥' }, ...(user.role === 'admin' ? [{ id: 'users' as View, label: 'User management', icon: '⚙' }] : []),
    { id: 'settings', label: 'Settings', icon: '◇' },
  ];

  const changeTheme = (next: Theme) => { setTheme(next); window.localStorage.setItem('ceit-theme', next); };

  return <main className={`app-shell theme-${theme} ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
    <aside className="sidebar">
      <div className="brand-lockup"><span>CEIT</span><div className="brand-copy">Computer Engineering &<small>Information Technology</small></div><button type="button" className="sidebar-toggle" aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'} aria-expanded={!sidebarCollapsed} onClick={() => setSidebarCollapsed((value) => !value)}>{sidebarCollapsed ? '›' : '‹'}</button></div>
      <nav aria-label="Main navigation">{nav.map((item) => <button key={item.id} title={sidebarCollapsed ? item.label : undefined} className={view === item.id ? 'active' : ''} onClick={() => { setView(item.id); setError(''); setNotice(''); }}><i>{item.icon}</i><span className="nav-label">{item.label}</span></button>)}</nav>
      <button className="account" onClick={signOut}><span>{user.full_name.slice(0, 1).toUpperCase()}</span><div><strong>{user.full_name}</strong><small>{user.role} · Sign out</small></div></button>
    </aside>
    <section className="main-panel">
      <header className="app-header"><div><p>MTU · CEIT Departmental Pilot</p><strong>{new Date().toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'long' })}</strong></div><div className="model-live"><span /> Synthetic ANN demo ready</div></header>
      {(error || notice) && <div className={`toast ${error ? 'error' : 'success'}`}><span>{error || notice}</span><button onClick={() => { setError(''); setNotice(''); }}>×</button></div>}
      {view === 'dashboard' && <DashboardView data={dashboard} setView={setView} />}
      {view === 'students' && <StudentsView students={students} onRefresh={refresh} setBusy={setBusy} busy={busy} onError={handleError} onNotice={setNotice} />}
      {view === 'predict' && <PredictView students={students} predictions={predictions} onRefresh={refresh} onError={handleError} />}
      {view === 'interventions' && <InterventionsView students={students} interventions={interventions} onRefresh={refresh} onError={handleError} onNotice={setNotice} />}
      {view === 'model' && <ModelView dashboard={dashboard} predictions={predictions} />}
      {view === 'users' && user.role === 'admin' && <UsersView onError={handleError} onNotice={setNotice} />}
      {view === 'settings' && <SettingsView theme={theme} onThemeChange={changeTheme} onError={handleError} onChanged={() => setUser(null)} />}
    </section>
  </main>;
}

function DashboardView({ data, setView }: { data: Dashboard | null; setView: (view: View) => void }) {
  if (!data) return <Empty title="Loading overview" text="Preparing the latest academic support summary." />;
  const totalRisk = Object.values(data.risk_distribution).reduce((a, b) => a + b, 0) || 1;
  return <div className="page-content">
    <div className="page-title"><div><h1>Overview</h1></div><button className="primary" onClick={() => setView('predict')}>New prediction <b>→</b></button></div>
    <section className="metrics"><Metric label="Registered students" value={data.summary.students} /><Metric label="High-risk signals" value={data.summary.high_risk} tone="risk" /><Metric label="Active support plans" value={data.summary.active_interventions} /><Metric label="UCI ANN accuracy" value={data.research?.models?.ANN?.accuracy ? `${Math.round(Number(data.research.models.ANN.accuracy) * 100)}%` : '—'} tone="accent" /></section>
    <section className="dashboard-grid">
      <article className="panel risk-panel"><div className="panel-title"><div><p className="kicker">Risk distribution</p><h2>Latest prediction signals</h2></div><span>{totalRisk} total</span></div>
        <div className="risk-visual"><div className="donut" style={{ background: `conic-gradient(#e2654a 0 ${(data.risk_distribution.high || 0) / totalRisk * 100}%, #e9b949 0 ${((data.risk_distribution.high || 0) + (data.risk_distribution.medium || 0)) / totalRisk * 100}%, #14845c 0)` }}><div><strong>{data.summary.high_risk}</strong><small>need attention</small></div></div>
          <ul><li><span className="dot high" />High risk <b>{data.risk_distribution.high || 0}</b></li><li><span className="dot medium" />Medium risk <b>{data.risk_distribution.medium || 0}</b></li><li><span className="dot low" />Low risk <b>{data.risk_distribution.low || 0}</b></li></ul></div>
      </article>
      <article className="panel"><div className="panel-title"><div><p className="kicker">Six-year cohort view</p><h2>Support signals by academic level</h2></div></div>
        {data.cohorts.length ? <div className="department-list">{data.cohorts.map((cohort) => <div key={`${cohort.academic_year}-${cohort.semester}`}><span><strong>{cohort.label}</strong><small>{cohort.students} students · {cohort.high_risk} high risk</small></span><span className="department-bar"><i style={{ width: `${cohort.avg_pass_probability || 0}%` }} /></span><b>{cohort.avg_pass_probability ?? '—'}%</b></div>)}</div> : <Empty title="No CEIT student records yet" text="Add students to compare academic levels." />}
      </article>
    </section>
    <section className="panel"><div className="panel-title"><div><p className="kicker">Recent activity</p><h2>Latest assessments</h2></div><button className="text-button" onClick={() => setView('predict')}>View prediction centre</button></div>
      {data.recent_predictions.length ? <div className="table-wrap"><table><thead><tr><th>Student</th><th>Department</th><th>Result</th><th>Probability</th><th>Risk</th><th>Date</th></tr></thead><tbody>{data.recent_predictions.map((item) => <tr key={item.id}><td><strong>{item.student_code || 'Ad-hoc'}</strong></td><td>{item.department || '—'}</td><td className="capitalize">{item.result_label || 'Grade estimate'}</td><td>{item.pass_probability !== undefined ? `${Math.round(item.pass_probability * 100)}%` : '—'}</td><td><RiskBadge risk={item.risk_level} /></td><td>{new Date(item.created_at).toLocaleDateString()}</td></tr>)}</tbody></table></div> : <Empty title="No predictions yet" text="Run the first assessment to populate this activity view." />}
    </section>
  </div>;
}

function StudentsView({ students, onRefresh, setBusy, busy, onError, onNotice }: { students: Student[]; onRefresh: () => Promise<void>; setBusy: (v: boolean) => void; busy: boolean; onError: (e: unknown) => void; onNotice: (s: string) => void }) {
  const [showForm, setShowForm] = useState(false); const [form, setForm] = useState(blankStudent); const [search, setSearch] = useState('');
  const filtered = students.filter((s) => `${s.student_code} ${s.department}`.toLowerCase().includes(search.toLowerCase()));
  async function submit(event: FormEvent) { event.preventDefault(); setBusy(true); try { await api('/api/students', { method: 'POST', body: JSON.stringify(form) }); setShowForm(false); setForm(blankStudent); onNotice('Student record added securely.'); await onRefresh(); } catch (e) { onError(e); } finally { setBusy(false); } }
  async function batch(file?: File) {
    if (!file) return;
    try {
      const text = await file.text(); const lines = text.trim().split(/\r?\n/); const headers = lines.shift()?.split(',').map((h) => h.trim()) ?? [];
      const required = ['student_code','academic_year','semester','gender','age','attendance','study_time','previous_grade','internet_access','family_support','absences','participation','homework_completion'];
      if (!required.every((field) => headers.includes(field))) throw new Error('CSV columns do not match the required template.');
      const records = lines.filter(Boolean).map((line) => { const values = line.split(',').map((v) => v.trim()); const raw = Object.fromEntries(headers.map((h, i) => [h, values[i]])); return { ...raw, department:CEIT, academic_year:+raw.academic_year, semester:+raw.semester, age:+raw.age, attendance:+raw.attendance, study_time:+raw.study_time, previous_grade:+raw.previous_grade, internet_access:['yes','true','1'].includes(raw.internet_access.toLowerCase()), family_support:['yes','true','1'].includes(raw.family_support.toLowerCase()), absences:+raw.absences, participation:+raw.participation, homework_completion:+raw.homework_completion }; });
      const result = await api<{ created: number; duplicates: string[] }>('/api/students/batch', { method: 'POST', body: JSON.stringify({ students: records }) });
      onNotice(`${result.created} student records imported${result.duplicates.length ? `; ${result.duplicates.length} duplicates skipped` : ''}.`); await onRefresh();
    } catch (e) { onError(e); }
  }
  return <div className="page-content"><div className="page-title"><div><h1>Students</h1></div><div className="button-row"><label className="secondary file-button">Import CSV<input type="file" accept=".csv,text/csv" onChange={(e) => batch(e.target.files?.[0])} /></label><button className="primary" onClick={() => setShowForm(!showForm)}>{showForm ? 'Close form' : 'Add student'} <b>+</b></button></div></div>
    {showForm && <form className="panel create-form" onSubmit={submit}><div className="panel-title"><div><p className="kicker">New CEIT record</p><h2>Academic profile</h2></div></div><div className="identity-grid"><label><span>Student code</span><input value={form.student_code} pattern="[A-Za-z0-9_-]+" onChange={(e) => setForm({ ...form, student_code: e.target.value })} required /></label><label><span>Major</span><input value="CEIT" disabled /></label><label><span>Academic level</span><select value={`${form.academic_year}-${form.semester}`} onChange={(e) => { const [year,semester]=e.target.value.split('-').map(Number); setForm({...form,academic_year:year,semester}); }}>{academicLevels.map((level)=><option key={`${level.year}-${level.semester}`} value={`${level.year}-${level.semester}`}>{level.label}</option>)}</select></label></div><FeatureFields value={form} onChange={(next) => setForm({ ...form, ...next })} /><div className="form-actions"><button className="secondary" type="button" onClick={() => setShowForm(false)}>Cancel</button><button className="primary" disabled={busy}>Save student securely</button></div></form>}
    <section className="panel"><div className="table-toolbar"><div><p className="kicker">CEIT directory</p><h2>{students.length} registered students</h2></div><input className="search" placeholder="Search student code…" value={search} onChange={(e) => setSearch(e.target.value)} /></div>
      {filtered.length ? <div className="table-wrap"><table><thead><tr><th>Student code</th><th>Major</th><th>Academic level</th><th>Attendance</th><th>Previous grade</th><th>Latest risk</th></tr></thead><tbody>{filtered.map((s) => <tr key={s.id}><td><strong>{s.student_code}</strong></td><td>CEIT</td><td>{s.academic_label || `Year ${s.academic_year}`}</td><td>{s.attendance}%</td><td>{s.previous_grade}</td><td><RiskBadge risk={s.risk_level} /></td></tr>)}</tbody></table></div> : <Empty title="No matching students" text="Add a record or change your search." />}
    </section>
  </div>;
}

function PredictView({ students, predictions, onRefresh, onError }: { students: Student[]; predictions: Prediction[]; onRefresh: () => Promise<void>; onError: (e: unknown) => void }) {
  const [studentId, setStudentId] = useState<number | ''>(''); const [features, setFeatures] = useState<Features>(blankFeatures);
  const [result, setResult] = useState<{ risk_level: string; result: { prediction?: string; pass_probability?: number; predicted_final_grade?: number; explanation?: Explanation[] } } | null>(null); const [loading, setLoading] = useState(false);
  async function submit(event: FormEvent) { event.preventDefault(); setLoading(true); setResult(null); try { const data = studentId ? await api<typeof result>(`/api/students/${studentId}/predict`, { method: 'POST' }) : await api<typeof result>('/api/predict', { method: 'POST', body: JSON.stringify(features) }); setResult(data); await onRefresh(); } catch (e) { onError(e); } finally { setLoading(false); } }
  return <div className="page-content"><div className="page-title"><div><h1>Prediction</h1></div></div>
    <section className="prediction-layout"><form className="panel prediction-form" onSubmit={submit}><div className="panel-title"><div><p className="kicker">Input profile</p><h2>Learning indicators</h2></div></div><label className="wide-label"><span>Registered student (optional)</span><select value={studentId} onChange={(e) => { const nextId = e.target.value ? +e.target.value : ''; setStudentId(nextId); const selected = students.find((s) => s.id === nextId); setFeatures(selected ?? blankFeatures); }}><option value="">Ad-hoc assessment — do not link to a record</option>{students.map((s) => <option key={s.id} value={s.id}>{s.student_code} · {s.department}</option>)}</select></label><FeatureFields value={features} onChange={setFeatures} /><div className="form-actions"><span className="local-note">● Processed locally</span><button className="primary" disabled={loading}>{loading ? 'Analysing…' : 'Run ANN assessment'} <b>→</b></button></div></form>
      <aside className="result-panel">{!result ? <div className="result-empty"><div className="signal-orb"><i /></div><p className="kicker">Model signal</p><h2>Ready to assess.</h2><p>Predictions indicate support needs. They never replace teacher judgment.</p></div> : <div className="result-content"><RiskBadge risk={result.risk_level} /><p className="kicker">Estimated outcome</p><strong className="big-score">{result.result.pass_probability !== undefined ? `${Math.round(result.result.pass_probability * 100)}%` : result.result.predicted_final_grade?.toFixed(1)}</strong><h2>{result.result.prediction ? `${result.result.prediction} likely` : 'Predicted final grade'}</h2><p>{result.result.pass_probability !== undefined ? 'Estimated probability of passing' : 'Estimated result on the source grade scale'}</p><div className="explanation"><h3>Strongest local influences</h3>{result.result.explanation?.map((item) => <div key={item.feature}><span>{item.label}</span><b className={item.impact >= 0 ? 'positive' : 'negative'}>{item.direction}</b></div>)}</div><div className="human-note"><strong>Human review required</strong>Offer guidance, tutoring or attendance support before making any academic decision.</div></div>}</aside>
    </section>
    <section className="panel"><div className="panel-title"><div><p className="kicker">Audit history</p><h2>Recent predictions</h2></div></div>{predictions.length ? <div className="table-wrap"><table><thead><tr><th>Student</th><th>Outcome</th><th>Pass probability</th><th>Risk</th><th>Reviewed by</th><th>Date</th></tr></thead><tbody>{predictions.slice(0, 20).map((p) => <tr key={p.id}><td><strong>{p.student_code || 'Ad-hoc'}</strong></td><td className="capitalize">{p.result_label || 'Grade estimate'}</td><td>{p.pass_probability !== undefined ? `${Math.round(p.pass_probability * 100)}%` : p.predicted_final_grade}</td><td><RiskBadge risk={p.risk_level} /></td><td>{p.created_by_name}</td><td>{new Date(p.created_at).toLocaleDateString()}</td></tr>)}</tbody></table></div> : <Empty title="No prediction history" text="Completed assessments appear here." />}</section>
  </div>;
}

function InterventionsView({ students, interventions, onRefresh, onError, onNotice }: { students: Student[]; interventions: Intervention[]; onRefresh: () => Promise<void>; onError: (e: unknown) => void; onNotice: (s: string) => void }) {
  const [show, setShow] = useState(false); const [form, setForm] = useState({ student_id: '', action: 'Academic advising session', assigned_to: '', due_date: '', notes: '' });
  async function add(event: FormEvent) { event.preventDefault(); try { await api('/api/interventions', { method: 'POST', body: JSON.stringify({ ...form, student_id: +form.student_id, due_date: form.due_date || null }) }); setShow(false); onNotice('Support plan created.'); await onRefresh(); } catch (e) { onError(e); } }
  async function update(item: Intervention, status: Intervention['status']) { try { await api(`/api/interventions/${item.id}`, { method: 'PATCH', body: JSON.stringify({ status, notes: item.notes }) }); onNotice('Intervention status updated.'); await onRefresh(); } catch (e) { onError(e); } }
  return <div className="page-content"><div className="page-title"><div><h1>Interventions</h1></div><button className="primary" onClick={() => setShow(!show)}>{show ? 'Close form' : 'Create support plan'} <b>+</b></button></div>
    {show && <form className="panel compact-form" onSubmit={add}><label><span>Student</span><select required value={form.student_id} onChange={(e) => setForm({ ...form, student_id: e.target.value })}><option value="">Select student</option>{students.map((s) => <option key={s.id} value={s.id}>{s.student_code} · {s.department}</option>)}</select></label><label><span>Support action</span><input value={form.action} onChange={(e) => setForm({ ...form, action: e.target.value })} required /></label><label><span>Assigned to</span><input placeholder="Advisor or department" value={form.assigned_to} onChange={(e) => setForm({ ...form, assigned_to: e.target.value })} required /></label><label><span>Due date</span><input type="date" value={form.due_date} onChange={(e) => setForm({ ...form, due_date: e.target.value })} /></label><label className="span-two"><span>Notes</span><textarea value={form.notes} onChange={(e) => setForm({ ...form, notes: e.target.value })} /></label><div className="form-actions span-two"><button className="primary">Save support plan</button></div></form>}
    <section className="kanban">{(['pending','in_progress','completed'] as const).map((status) => <div className="kanban-column" key={status}><header><h2>{status.replace('_',' ')}</h2><span>{interventions.filter((i) => i.status === status).length}</span></header>{interventions.filter((i) => i.status === status).map((item) => <article key={item.id}><div><strong>{item.student_code}</strong><RiskBadge risk={students.find((s) => s.id === item.student_id)?.risk_level} /></div><h3>{item.action}</h3><p>{item.notes || 'No additional notes.'}</p><small>{item.assigned_to}{item.due_date ? ` · Due ${item.due_date}` : ''}</small><select aria-label="Update status" value={item.status} onChange={(e) => update(item, e.target.value as Intervention['status'])}><option value="pending">Pending</option><option value="in_progress">In progress</option><option value="completed">Completed</option></select></article>)}{!interventions.some((i) => i.status === status) && <p className="column-empty">No items</p>}</div>)}</section>
  </div>;
}

function ModelView({ dashboard, predictions }: { dashboard: Dashboard | null; predictions: Prediction[] }) {
  const research = dashboard?.research; const models = research?.models ?? {}; const annGaps = research?.fairness_gaps?.ANN ?? {};
  const operational = dashboard?.model_metrics ?? {}; const metadata = dashboard?.model_metadata;
  const percentage = (value: unknown) => typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : '—';
  return <div className="page-content"><div className="page-title"><div><h1>Model & reports</h1></div><div className="button-row"><a className="secondary link-button" href={`${API_URL}/api/research/files/model_comparison.csv`} target="_blank" rel="noreferrer">Results CSV ↓</a><a className="primary link-button" href={`${API_URL}/api/reports/at-risk.csv`} target="_blank" rel="noreferrer">At-risk CSV ↓</a></div></div>
    <div className="validation-status"><span>MTU CEIT validation</span><strong>Awaiting approved anonymized dataset</strong><p>The UCI results below establish the experiment pipeline; they do not establish accuracy for MTU students.</p></div>
    <div className="synthetic-status"><strong>Operational demo ANN · synthetic dataset</strong><span>{metadata?.training_rows ?? '—'} training + {metadata?.validation_rows ?? '—'} validation + {metadata?.test_rows ?? '—'} test records</span><span>Accuracy {percentage(operational.accuracy)} · majority baseline {percentage(operational.majority_baseline_accuracy)}</span></div>
    <section className="metrics"><Metric label="UCI records" value={research?.source_rows ?? '—'} /><Metric label="ANN accuracy" value={percentage(models.ANN?.accuracy)} /><Metric label="ANN F1 score" value={models.ANN?.f1_score ? Number(models.ANN.f1_score).toFixed(3) : '—'} /><Metric label="Best F1 model" value={research?.best_model_by_f1 ?? '—'} tone="accent" /></section>
    <section className="panel research-panel"><div className="panel-title"><div><p className="kicker">Baseline comparison</p><h2>ANN vs conventional models</h2></div><span>Fixed seed · stratified split</span></div>
      <div className="table-wrap"><table><thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 score</th><th>ROC AUC</th></tr></thead><tbody>{Object.entries(models).map(([name, values]) => <tr key={name} className={name === research?.best_model_by_f1 ? 'best-row' : ''}><td><strong>{name}</strong>{name === research?.best_model_by_f1 && <small className="best-label">Best F1</small>}</td><td>{percentage(values.accuracy)}</td><td>{percentage(values.precision)}</td><td>{percentage(values.recall)}</td><td>{Number(values.f1_score).toFixed(3)}</td><td>{Number(values.roc_auc).toFixed(3)}</td></tr>)}</tbody></table></div>
      <img className="research-chart" src={`${API_URL}/api/research/assets/model_comparison.png`} alt="Bar chart comparing ANN, Logistic Regression and Random Forest evaluation metrics" />
    </section>
    <section className="dashboard-grid"><article className="panel"><div className="panel-title"><div><p className="kicker">ANN architecture</p><h2>Neural network design</h2></div></div><div className="architecture"><div><strong>Input</strong><span>9 available indicators</span></div><i>→</i><div><strong>64</strong><span>ReLU + dropout</span></div><i>→</i><div><strong>32</strong><span>ReLU + dropout</span></div><i>→</i><div><strong>1</strong><span>Sigmoid output</span></div></div><ul className="check-list"><li>Training-only preprocessing prevents test leakage</li><li>Early stopping and L2 regularization limit overfitting</li><li>Same held-out records used for all three models</li><li>Homework completion is unavailable in UCI and is explicitly excluded</li></ul></article>
      <article className="panel"><div className="panel-title"><div><p className="kicker">Bias audit</p><h2>ANN group-gap screening</h2></div><a className="text-button" href={`${API_URL}/api/research/files/fairness_audit.csv`} target="_blank" rel="noreferrer">Download detail</a></div><div className="gap-grid"><div><span>Gender accuracy gap</span><strong>{percentage(annGaps.gender_accuracy_gap)}</strong></div><div><span>Gender recall gap</span><strong>{percentage(annGaps.gender_recall_gap)}</strong></div><div><span>Family support accuracy gap</span><strong>{percentage(annGaps.family_support_accuracy_gap)}</strong></div><div><span>Family support recall gap</span><strong>{percentage(annGaps.family_support_recall_gap)}</strong></div></div><p className="bias-note">Descriptive gaps on a small UCI test set are warning indicators, not proof of discrimination or causation. Review group sample sizes and false-positive rates before use.</p></article></section>
    <section className="panel research-panel"><div className="panel-title"><div><p className="kicker">Evaluation detail</p><h2>Confusion matrices & fairness graph</h2></div></div><div className="research-images"><img src={`${API_URL}/api/research/assets/confusion_matrices.png`} alt="Confusion matrices for all benchmark models" /><img src={`${API_URL}/api/research/assets/fairness_audit.png`} alt="Fairness metric gaps by gender and family support" /></div></section>
    <section className="panel governance"><div><p className="kicker">Responsible use</p><h2>Required institutional safeguards</h2></div><div className="governance-grid"><article><span>01</span><strong>Human oversight</strong><p>Never use predictions as the sole basis for grading, discipline or denying opportunity.</p></article><article><span>02</span><strong>Bias review</strong><p>Repeat group evaluation using approved and representative MTU CEIT data.</p></article><article><span>03</span><strong>Data minimisation</strong><p>Use student codes, limit retention and grant access only to authorized CEIT staff.</p></article><article><span>04</span><strong>Continuous monitoring</strong><p>Review drift, document consent and version every production model.</p></article></div><p className="report-count">This local system currently stores {predictions.length} audited prediction records.</p></section>
  </div>;
}

function UsersView({ onError, onNotice }: { onError: (e: unknown) => void; onNotice: (s: string) => void }) {
  const [users, setUsers] = useState<(User & { is_active: number; created_at: string })[]>([]); const [show, setShow] = useState(false);
  const [form, setForm] = useState({ email: '', full_name: '', role: 'teacher', password: '' });
  const load = useCallback(() => api<{ users: typeof users }>('/api/users').then((data) => setUsers(data.users)).catch(onError), [onError]);
  useEffect(() => { load(); }, [load]);
  async function submit(event: FormEvent) { event.preventDefault(); try { await api('/api/users', { method: 'POST', body: JSON.stringify(form) }); setShow(false); setForm({ email:'', full_name:'', role:'teacher', password:'' }); onNotice('Authorized user created.'); load(); } catch (e) { onError(e); } }
  return <div className="page-content"><div className="page-title"><div><h1>Users</h1></div><button className="primary" onClick={() => setShow(!show)}>{show ? 'Close form' : 'Add user'} <b>+</b></button></div>
    {show && <form className="panel compact-form" onSubmit={submit}><label><span>Full name</span><input value={form.full_name} onChange={(e) => setForm({...form,full_name:e.target.value})} required /></label><label><span>Email</span><input type="email" value={form.email} onChange={(e) => setForm({...form,email:e.target.value})} required /></label><label><span>Role</span><select value={form.role} onChange={(e) => setForm({...form,role:e.target.value})}><option value="teacher">Teacher</option><option value="admin">Administrator</option></select></label><label><span>Temporary password</span><input type="password" minLength={12} value={form.password} onChange={(e) => setForm({...form,password:e.target.value})} required /></label><div className="form-actions span-two"><button className="primary">Create authorized user</button></div></form>}
    <section className="panel"><div className="panel-title"><div><p className="kicker">Role directory</p><h2>{users.length} authorized accounts</h2></div></div><div className="user-grid">{users.map((item) => <article key={item.id}><span>{item.full_name[0]}</span><div><strong>{item.full_name}</strong><p>{item.email}</p></div><b>{item.role}</b></article>)}</div></section>
  </div>;
}

function SettingsView({ theme, onThemeChange, onError, onChanged }: { theme: Theme; onThemeChange: (theme: Theme) => void; onError: (e: unknown) => void; onChanged: () => void }) {
  const [currentPassword, setCurrentPassword] = useState(''); const [newPassword, setNewPassword] = useState(''); const [confirm, setConfirm] = useState('');
  async function submit(event: FormEvent) { event.preventDefault(); if (newPassword !== confirm) { onError(new Error('New passwords do not match.')); return; } try { await api('/api/auth/change-password', { method:'POST', body:JSON.stringify({ current_password:currentPassword, new_password:newPassword }) }); onChanged(); } catch (e) { onError(e); } }
  return <div className="page-content"><div className="page-title"><div><h1>Settings</h1></div></div>
    <section className="panel appearance-card"><h2>Theme</h2><div className="theme-options" role="radiogroup" aria-label="Color theme">{themeOptions.map((option) => <button type="button" role="radio" aria-checked={theme === option.id} className={`theme-choice ${theme === option.id ? 'active' : ''}`} key={option.id} onClick={() => onThemeChange(option.id)}><span style={{ background:option.color }} />{option.label}</button>)}</div></section>
    <section className="panel settings-card"><div><h2>Change password</h2><p>All active sessions will be signed out.</p></div><form onSubmit={submit}><label><span>Current password</span><input type="password" autoComplete="current-password" value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} required /></label><label><span>New password</span><input type="password" autoComplete="new-password" minLength={12} value={newPassword} onChange={(e) => setNewPassword(e.target.value)} required /></label><label><span>Confirm new password</span><input type="password" autoComplete="new-password" minLength={12} value={confirm} onChange={(e) => setConfirm(e.target.value)} required /></label><button className="primary">Update password</button></form></section>
  </div>;
}
