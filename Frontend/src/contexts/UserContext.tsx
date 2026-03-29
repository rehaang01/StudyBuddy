// ─────────────────────────────────────────────────────────────────────────────
// UserContext.tsx — Multi-user profile management for StudyBuddy
//
// Architecture:
//   • Provides currentUser, allUsers, switchUser, profileNames, setProfileName,
//     and getDisplayName to the entire app.
//   • Active profile is persisted to sessionStorage (sb_active_user).
//   • profileNames is persisted to localStorage (sb_profile_names) so it is
//     available SYNCHRONOUSLY on the first render — eliminating name flicker.
//   • On startup, names are read from localStorage immediately (no async gap),
//     then a background fetch refreshes them from Cosmos and re-persists.
//   • setProfileName updates both React state and localStorage instantly after
//     a Settings save, so no second fetch is ever needed.
// ─────────────────────────────────────────────────────────────────────────────

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { USER_PROFILES, DEFAULT_USER, type UserProfile } from "@/config/users";
import { clearAPICache } from "@/lib/offlineStore";
import { API_BASE } from "@/config/api";

// ── Storage keys ──────────────────────────────────────────────────────────────

const SESSION_KEY = "sb_active_user";
const NAMES_KEY   = "sb_profile_names"; // localStorage — survives refresh

// ── Helpers ───────────────────────────────────────────────────────────────────

function getInitialUser(): UserProfile {
  try {
    const stored = sessionStorage.getItem(SESSION_KEY);
    if (stored) {
      const found = USER_PROFILES.find((u) => u.id === stored);
      if (found) return found;
    }
  } catch {
    // sessionStorage blocked
  }
  return DEFAULT_USER;
}

/** Read the persisted names map from localStorage synchronously.
 *  Returns {} on any failure so callers always get a safe object. */
function readPersistedNames(): Record<string, string> {
  try {
    const raw = localStorage.getItem(NAMES_KEY);
    if (raw) return JSON.parse(raw) as Record<string, string>;
  } catch {
    // localStorage blocked or JSON malformed
  }
  return {};
}

/** Persist the names map to localStorage so it survives page refreshes. */
function persistNames(map: Record<string, string>): void {
  try {
    localStorage.setItem(NAMES_KEY, JSON.stringify(map));
  } catch {
    // localStorage blocked — silent
  }
}

// ── Context shape ─────────────────────────────────────────────────────────────

export interface UserContextValue {
  currentUser: UserProfile;
  allUsers: readonly UserProfile[];
  switchUser: (id: string) => void;
  profileNames: Record<string, string>;
  setProfileName: (userId: string, name: string) => void;
  /** Best name for a user: custom DB name → static displayName fallback. */
  getDisplayName: (userId: string) => string;
}

const UserContext = createContext<UserContextValue | null>(null);

// ── Provider ──────────────────────────────────────────────────────────────────

export function UserProvider({ children }: { children: ReactNode }) {
  const [currentUser, setCurrentUser] = useState<UserProfile>(getInitialUser);

  // ── Initialise from localStorage — zero flicker ───────────────────────────
  // readPersistedNames() is synchronous, so the very first render already has
  // the correct names. No "Rehaan → john" flash.
  const [profileNames, setProfileNames] = useState<Record<string, string>>(
    readPersistedNames
  );

  // ── Background refresh from Cosmos on mount ───────────────────────────────
  // Keeps names in sync if another device made a change since last visit.
  useEffect(() => {
    const refresh = async () => {
      try {
        const results = await Promise.all(
          USER_PROFILES.map((u) =>
            fetch(`${API_BASE}/settings/?user_id=${u.id}`)
              .then((r) => (r.ok ? r.json() : null))
              .then((data) => ({
                id: u.id,
                name: (data?.profile?.display_name as string) ?? "",
              }))
              .catch(() => ({ id: u.id, name: "" }))
          )
        );
        const map: Record<string, string> = {};
        results.forEach(({ id, name }) => { map[id] = name; });
        setProfileNames(map);
        persistNames(map);
      } catch {
        // Network failure — cached names remain, silent
      }
    };
    refresh();
  }, []); // once on mount

  // ── setProfileName — instant update after Settings save ──────────────────
  const setProfileName = useCallback((userId: string, name: string) => {
    setProfileNames((prev) => {
      const next = { ...prev, [userId]: name };
      persistNames(next); // write through to localStorage immediately
      return next;
    });
  }, []);

  // ── getDisplayName — convenience helper ───────────────────────────────────
  const getDisplayName = useCallback(
    (userId: string): string => {
      const custom = profileNames[userId];
      if (custom) return custom;
      return USER_PROFILES.find((u) => u.id === userId)?.displayName ?? "";
    },
    [profileNames]
  );

  // ── switchUser ────────────────────────────────────────────────────────────
  const switchUser = useCallback(
    (id: string) => {
      if (id === currentUser.id) return;
      const found = USER_PROFILES.find((u) => u.id === id);
      if (!found) return;
      try { sessionStorage.setItem(SESSION_KEY, id); } catch { /* blocked */ }
      setCurrentUser(found);
    },
    [currentUser.id]
  );

  const value = useMemo<UserContextValue>(
    () => ({
      currentUser,
      allUsers: USER_PROFILES,
      switchUser,
      profileNames,
      setProfileName,
      getDisplayName,
    }),
    [currentUser, switchUser, profileNames, setProfileName, getDisplayName]
  );

  return <UserContext.Provider value={value}>{children}</UserContext.Provider>;
}

// ── Hook ──────────────────────────────────────────────────────────────────────

export function useUser(): UserContextValue {
  const ctx = useContext(UserContext);
  if (!ctx) throw new Error("useUser must be used within a UserProvider");
  return ctx;
}
