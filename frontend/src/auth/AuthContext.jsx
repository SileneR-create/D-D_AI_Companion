/**
 * Contexte d'authentification : expose l'utilisateur courant et les actions
 * login / register / logout a toute l'application.
 */
import { createContext, useCallback, useContext, useEffect, useState } from "react";
import * as authApi from "../api/auth.js";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Au montage : restaure la session si un jeton valide existe.
  useEffect(() => {
    authApi.me().then(setUser).finally(() => setLoading(false));
  }, []);

  const login = useCallback(async (credentials) => {
    await authApi.login(credentials);
    setUser(await authApi.me());
  }, []);

  const register = useCallback(async (data) => {
    await authApi.register(data);
    await authApi.login({ username: data.username, password: data.password });
    setUser(await authApi.me());
  }, []);

  const logout = useCallback(() => {
    authApi.logout();
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth doit etre utilise dans <AuthProvider>");
  return ctx;
}
