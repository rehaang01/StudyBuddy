/**
 * coinStore.ts — Shared gamification types/constants plus legacy localStorage helpers.
 */

export function getTodayIST(): string {
  const now = new Date();
  const ist = new Date(now.getTime() + (5.5 * 60 * 60 * 1000));
  return ist.toISOString().slice(0, 10);
}

function getYesterdayIST(): string {
  const now = new Date();
  const ist = new Date(now.getTime() + (5.5 * 60 * 60 * 1000) - 86400000);
  return ist.toISOString().slice(0, 10);
}

export interface CoinTransaction {
  id: string; type: "earn" | "spend"; amount: number;
  reason: string; category: string; timestamp: string;
}
export interface StoreOrder {
  id: string; item_id: string; item_name: string; cost: number;
  ordered_at: string; status: "delivered" | "pending";
}
export interface MissionProgress {
  mission_id: string; completed: boolean; completed_at?: string;
}
export interface CoinState {
  balance: number; lifetime_earned: number;
  login_streak: number; longest_streak: number;
  last_login_date: string | null; last_reward_date: string | null;
  transactions: CoinTransaction[]; orders: StoreOrder[];
  missions: Record<string, MissionProgress>;
  referral_code: string; referred_by: string | null; referral_count: number;
}

export const LEGACY_COIN_STORAGE_KEY = "studybuddy_coins";
/** Latest serialized shape for legacy browser-side coin data. */
const ECONOMY_VERSION = 2;

export const REWARDS = {
  DAILY_LOGIN: 2,
  STREAK_30: 30, STREAK_90: 75, STREAK_365: 200,
  QUIZ_COMPLETE: 3, DOCUMENT_UPLOAD: 2,
  REFERRAL_SENDER: 15, REFERRAL_RECEIVER: 10,
} as const;

// ── Store Items ─────────────────────────────────────────────────────────────
export interface StoreItem {
  id: string; name: string; description: string; cost: number;
  category: "boost" | "cosmetic"; iconKey: string; gradient: string; limited?: boolean;
}

export const STORE_ITEMS: StoreItem[] = [
  { id: "extra_upload", name: "Extra Upload Slot", description: "Upload one additional document beyond the 5-file limit", cost: 180, category: "boost", iconKey: "file-plus", gradient: "from-primary/15 to-blue-500/15" },
  { id: "ai_deep_dive", name: "AI Deep Dive", description: "Unlock extended AI analysis with extra-long context for one session", cost: 250, category: "boost", iconKey: "brain", gradient: "from-violet-500/15 to-purple-500/15" },
  { id: "quiz_master", name: "Quiz Master Pack", description: "Generate up to 25 questions per quiz instead of the default 10", cost: 150, category: "boost", iconKey: "clipboard-list", gradient: "from-cyan-500/15 to-blue-500/15" },
  { id: "theme_midnight", name: "Midnight Scholar Theme", description: "Exclusive deep purple accent theme for your StudyBuddy interface", cost: 300, category: "cosmetic", iconKey: "moon", gradient: "from-indigo-500/15 to-violet-500/15" },
  { id: "theme_forest", name: "Forest Focus Theme", description: "Calming forest green accent theme for distraction-free study", cost: 300, category: "cosmetic", iconKey: "tree-pine", gradient: "from-emerald-500/15 to-green-500/15" },
  { id: "badge_scholar", name: "Scholar Badge", description: "Display an exclusive Scholar badge on your profile", cost: 450, category: "cosmetic", iconKey: "graduation-cap", gradient: "from-primary/15 to-sky-500/15", limited: true },
  { id: "tts_pack", name: "Extended TTS Pack", description: "Unlock 3 additional premium voice styles for text-to-speech", cost: 280, category: "boost", iconKey: "volume-2", gradient: "from-teal-500/15 to-cyan-500/15" },
  { id: "study_streak_shield", name: "Streak Shield", description: "Protect your study streak — miss one day without breaking it", cost: 140, category: "boost", iconKey: "shield", gradient: "from-primary/15 to-indigo-500/15", limited: true },
];

// ── Earn Missions (no achievements, no graph equation, no 7-day streak) ────
export interface EarnMission {
  id: string; name: string; description: string; reward: number;
  category: "daily" | "streak" | "social"; iconKey: string; repeatable: boolean;
}

export const EARN_MISSIONS: EarnMission[] = [
  { id: "daily_login", name: "Daily Login", description: "Open StudyBuddy once a day", reward: REWARDS.DAILY_LOGIN, category: "daily", iconKey: "log-in", repeatable: true },
  { id: "complete_quiz", name: "Complete a Quiz", description: "Finish any quiz with all questions answered", reward: REWARDS.QUIZ_COMPLETE, category: "daily", iconKey: "clipboard-check", repeatable: true },
  { id: "upload_doc", name: "Upload a Document", description: "Upload and process a new study document", reward: REWARDS.DOCUMENT_UPLOAD, category: "daily", iconKey: "upload", repeatable: true },
  { id: "streak_30", name: "30-Day Streak", description: "Log in for 30 consecutive days", reward: REWARDS.STREAK_30, category: "streak", iconKey: "flame", repeatable: false },
  { id: "streak_90", name: "90-Day Streak", description: "Log in for 90 consecutive days", reward: REWARDS.STREAK_90, category: "streak", iconKey: "star", repeatable: false },
  { id: "streak_365", name: "365-Day Streak", description: "Log in every day for a year", reward: REWARDS.STREAK_365, category: "streak", iconKey: "trophy", repeatable: false },
  { id: "referral", name: "Refer a Friend", description: "Share your code and get a friend to join", reward: REWARDS.REFERRAL_SENDER, category: "social", iconKey: "users", repeatable: true },
];

// ── Core ────────────────────────────────────────────────────────────────────
function genId(): string { return crypto.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`; }
function genCode(): string { const c = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"; let s = "SB-"; for (let i = 0; i < 6; i++) s += c[Math.floor(Math.random() * c.length)]; return s; }
export function createDefaultCoinState(): CoinState { return { balance: 0, lifetime_earned: 0, login_streak: 0, longest_streak: 0, last_login_date: null, last_reward_date: null, transactions: [], orders: [], missions: {}, referral_code: genCode(), referred_by: null, referral_count: 0 }; }

function coerceInt(value: unknown, fallback = 0): number {
  const parsed = Number.parseInt(String(value ?? ""), 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function normalizeLegacyCoinState(raw: unknown): CoinState | null {
  if (!raw || typeof raw !== "object") return null;
  const parsed = raw as Record<string, unknown>;

  const looksLikeCoinState = [
    "balance",
    "lifetime_earned",
    "login_streak",
    "last_login_date",
    "transactions",
    "referral_code",
  ].some((key) => key in parsed);

  if (!looksLikeCoinState) return null;

  return {
    balance: Math.max(0, coerceInt(parsed.balance, 0)),
    lifetime_earned: Math.max(
      Math.max(0, coerceInt(parsed.balance, 0)),
      coerceInt(parsed.lifetime_earned, 0),
    ),
    login_streak: Math.max(0, coerceInt(parsed.login_streak, 0)),
    longest_streak: Math.max(
      Math.max(0, coerceInt(parsed.login_streak, 0)),
      coerceInt(parsed.longest_streak, 0),
    ),
    last_login_date: typeof parsed.last_login_date === "string" ? parsed.last_login_date : null,
    last_reward_date: typeof parsed.last_reward_date === "string" ? parsed.last_reward_date : null,
    transactions: Array.isArray(parsed.transactions) ? parsed.transactions as CoinTransaction[] : [],
    orders: Array.isArray(parsed.orders) ? parsed.orders as StoreOrder[] : [],
    missions: (parsed.missions && typeof parsed.missions === "object") ? parsed.missions as Record<string, MissionProgress> : {},
    referral_code: typeof parsed.referral_code === "string" && parsed.referral_code ? parsed.referral_code : genCode(),
    referred_by: typeof parsed.referred_by === "string" ? parsed.referred_by : null,
    referral_count: Math.max(0, coerceInt(parsed.referral_count, 0)),
  };
}

export function hasMeaningfulCoinState(state: CoinState | null | undefined): state is CoinState {
  if (!state) return false;
  return (
    state.balance > 0 ||
    state.lifetime_earned > 0 ||
    state.login_streak > 0 ||
    state.longest_streak > 0 ||
    !!state.last_login_date ||
    !!state.last_reward_date ||
    state.transactions.length > 0 ||
    state.orders.length > 0 ||
    Object.keys(state.missions).length > 0 ||
    !!state.referred_by ||
    state.referral_count > 0
  );
}

export function getLegacyCoinState(): CoinState | null {
  try {
    const raw = localStorage.getItem(LEGACY_COIN_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    const normalized = normalizeLegacyCoinState(parsed);
    if (!normalized) {
      localStorage.removeItem(LEGACY_COIN_STORAGE_KEY);
      return null;
    }
    if (parsed._v !== ECONOMY_VERSION) {
      save(normalized);
    }
    return normalized;
  } catch {
    return null;
  }
}

export function clearLegacyCoinState(): void {
  localStorage.removeItem(LEGACY_COIN_STORAGE_KEY);
}

export function getCoinState(): CoinState {
  try {
    return getLegacyCoinState() ?? createDefaultCoinState();
  } catch { return createDefaultCoinState(); }
}
function save(s: CoinState) { localStorage.setItem(LEGACY_COIN_STORAGE_KEY, JSON.stringify({ ...s, _v: ECONOMY_VERSION })); }

export function earnCoins(amount: number, reason: string, category: string): CoinState {
  const s = getCoinState(); s.balance += amount; s.lifetime_earned += amount;
  s.transactions.unshift({ id: genId(), type: "earn", amount, reason, category, timestamp: new Date().toISOString() });
  if (s.transactions.length > 200) s.transactions = s.transactions.slice(0, 200); save(s); return s;
}

export function spendCoins(amount: number, reason: string, category: string): CoinState | null {
  const s = getCoinState(); if (s.balance < amount) return null; s.balance -= amount;
  s.transactions.unshift({ id: genId(), type: "spend", amount, reason, category, timestamp: new Date().toISOString() });
  if (s.transactions.length > 200) s.transactions = s.transactions.slice(0, 200); save(s); return s;
}

export function processDailyLogin(): { coins_earned: number; new_streak: number; streak_bonus: number; streak_milestone: string | null } | null {
  const s = getCoinState(); const today = getTodayIST(); const yesterday = getYesterdayIST();
  if (s.last_reward_date === today) return null;
  let ns = 1; if (s.last_login_date === yesterday) ns = s.login_streak + 1; else if (s.last_login_date === today) ns = s.login_streak;
  s.login_streak = ns; s.longest_streak = Math.max(s.longest_streak, ns); s.last_login_date = today; s.last_reward_date = today;
  let total = REWARDS.DAILY_LOGIN, sb = 0; let sm: string | null = null;
  if (ns === 30 && !s.missions["streak_30"]?.completed) { sb = REWARDS.STREAK_30; sm = "30-Day Streak"; s.missions["streak_30"] = { mission_id: "streak_30", completed: true, completed_at: today }; }
  else if (ns === 90 && !s.missions["streak_90"]?.completed) { sb = REWARDS.STREAK_90; sm = "90-Day Streak"; s.missions["streak_90"] = { mission_id: "streak_90", completed: true, completed_at: today }; }
  else if (ns === 365 && !s.missions["streak_365"]?.completed) { sb = REWARDS.STREAK_365; sm = "365-Day Streak"; s.missions["streak_365"] = { mission_id: "streak_365", completed: true, completed_at: today }; }
  total += sb; s.balance += total; s.lifetime_earned += total;
  s.transactions.unshift({ id: genId(), type: "earn", amount: total, reason: sm ? `Daily login + ${sm}` : `Daily login (Day ${ns})`, category: "login", timestamp: new Date().toISOString() });
  s.missions["daily_login"] = { mission_id: "daily_login", completed: true, completed_at: today };
  if (s.transactions.length > 200) s.transactions = s.transactions.slice(0, 200); save(s);
  return { coins_earned: REWARDS.DAILY_LOGIN, new_streak: ns, streak_bonus: sb, streak_milestone: sm };
}

export function purchaseItem(item: StoreItem): StoreOrder | null {
  const r = spendCoins(item.cost, `Purchased: ${item.name}`, "store"); if (!r) return null;
  const o: StoreOrder = { id: genId(), item_id: item.id, item_name: item.name, cost: item.cost, ordered_at: new Date().toISOString(), status: "delivered" };
  const s = getCoinState(); s.orders.unshift(o); save(s); return o;
}

export function applyReferralCode(code: string): boolean {
  const s = getCoinState(); if (code === s.referral_code || s.referred_by) return false;
  s.referred_by = code; s.balance += REWARDS.REFERRAL_RECEIVER; s.lifetime_earned += REWARDS.REFERRAL_RECEIVER;
  s.transactions.unshift({ id: genId(), type: "earn", amount: REWARDS.REFERRAL_RECEIVER, reason: `Referral bonus — code ${code}`, category: "referral", timestamp: new Date().toISOString() });
  save(s); return true;
}

export function recordReferralUsed(): void {
  const s = getCoinState(); s.referral_count++; s.balance += REWARDS.REFERRAL_SENDER; s.lifetime_earned += REWARDS.REFERRAL_SENDER;
  s.transactions.unshift({ id: genId(), type: "earn", amount: REWARDS.REFERRAL_SENDER, reason: "Friend used your referral code", category: "referral", timestamp: new Date().toISOString() });
  save(s);
}

export function completeMission(missionId: string): number {
  const m = EARN_MISSIONS.find(x => x.id === missionId); if (!m) return 0;
  const s = getCoinState(); const ex = s.missions[missionId];
  if (!m.repeatable && ex?.completed) return 0;
  if (m.repeatable && ex?.completed_at === getTodayIST()) return 0;
  s.missions[missionId] = { mission_id: missionId, completed: true, completed_at: getTodayIST() };
  s.balance += m.reward; s.lifetime_earned += m.reward;
  s.transactions.unshift({ id: genId(), type: "earn", amount: m.reward, reason: `Mission: ${m.name}`, category: "mission", timestamp: new Date().toISOString() });
  if (s.transactions.length > 200) s.transactions = s.transactions.slice(0, 200); save(s); return m.reward;
}

export function isMissionCompletedToday(missionId: string, state: CoinState = getCoinState()): boolean {
  const m = state.missions[missionId]; return !!m?.completed && m.completed_at === getTodayIST();
}

export function getBalance(): number { return getCoinState().balance; }
