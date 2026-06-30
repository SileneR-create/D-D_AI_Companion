/** Ecran de connexion / inscription, au theme Grimoire. */
import { useState } from "react";
import { LogIn, UserPlus } from "lucide-react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";
import { Crest } from "../components/atmosphere.jsx";
import { Divider } from "../components/ornaments.jsx";
import { useAuth } from "../auth/AuthContext.jsx";

export function AuthScreen() {
  const { login, register } = useAuth();
  const [mode, setMode] = useState("login"); // login | register
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("");
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const submit = async () => {
    setError(null);
    if (!username.trim() || !password) {
      setError("Indiquez un nom et un mot de passe.");
      return;
    }
    setBusy(true);
    try {
      if (mode === "login") await login({ username, password });
      else await register({ username, password, email });
    } catch (e) {
      setError(e?.message || "Une erreur est survenue.");
    } finally {
      setBusy(false);
    }
  };

  const isLogin = mode === "login";

  return (
    <div style={{ minHeight: "100vh", display: "flex", alignItems: "center", justifyContent: "center", padding: 24 }}>
      <div style={{
        width: 380, padding: "34px 32px", textAlign: "center",
        background: `linear-gradient(180deg, ${T.panel2}, ${T.panel})`,
        border: `1px solid ${T.line}`, borderRadius: 6, boxShadow: "0 18px 50px rgba(0,0,0,.55)",
      }}>
        <div style={{ display: "flex", justifyContent: "center" }}><Crest size={72} /></div>
        <div style={{ fontFamily: DISPLAY, fontSize: 11, letterSpacing: 5, textTransform: "uppercase", color: T.gold, marginTop: 14 }}>
          Compagnon des Royaumes
        </div>
        <h1 style={{ fontFamily: ORNATE, fontSize: 26, color: T.parch, margin: "6px 0 0" }}>
          {isLogin ? "Franchir le seuil" : "Sceller un pacte"}
        </h1>

        <div style={{ margin: "18px 0" }}><Divider /></div>

        <Field label="Nom" value={username} onChange={setUsername} onEnter={submit} />
        {!isLogin && <Field label="Courriel (optionnel)" value={email} onChange={setEmail} onEnter={submit} />}
        <Field label="Mot de passe" value={password} onChange={setPassword} onEnter={submit} type="password" />

        {error && <div style={{ fontFamily: BODY, fontSize: 14, color: T.emberBright, marginTop: 10 }}>{error}</div>}

        <button onClick={submit} disabled={busy} className="glow" style={{
          width: "100%", marginTop: 18, padding: "11px 0", borderRadius: 3,
          cursor: busy ? "wait" : "pointer", opacity: busy ? 0.6 : 1,
          display: "inline-flex", alignItems: "center", justifyContent: "center", gap: 8,
          background: "transparent", border: `1px solid ${T.gold}`, color: T.gold,
          fontFamily: DISPLAY, fontSize: 12, letterSpacing: 2, textTransform: "uppercase",
        }}>
          {isLogin ? <LogIn size={15} /> : <UserPlus size={15} />}
          {isLogin ? "Entrer" : "Creer mon compte"}
        </button>

        <button onClick={() => { setMode(isLogin ? "register" : "login"); setError(null); }} style={{
          marginTop: 16, background: "none", border: "none", cursor: "pointer",
          fontFamily: BODY, fontStyle: "italic", fontSize: 14.5, color: T.mistDim,
        }}>
          {isLogin ? "Pas encore de pacte ? Creer un compte" : "Deja un compte ? Se connecter"}
        </button>
      </div>
    </div>
  );
}

function Field({ label, value, onChange, onEnter, type = "text" }) {
  return (
    <div style={{ textAlign: "left", marginBottom: 12 }}>
      <label style={{ fontFamily: DISPLAY, fontSize: 9.5, letterSpacing: 2, textTransform: "uppercase", color: T.mistDim }}>{label}</label>
      <input type={type} value={value} onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && onEnter()}
        style={{ width: "100%", marginTop: 5, padding: "10px 12px", borderRadius: 3,
          background: T.ink, border: `1px solid ${T.line}`, color: T.parch,
          fontFamily: BODY, fontSize: 16, outline: "none", boxSizing: "border-box" }} />
    </div>
  );
}
