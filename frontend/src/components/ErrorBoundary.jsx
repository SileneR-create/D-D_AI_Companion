/** Empeche un crash de rendu de blanchir toute l'application. */
import { Component } from "react";
import { T, DISPLAY, ORNATE, BODY } from "../theme.js";

export class ErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { error: null }; }
  static getDerivedStateFromError(error) { return { error }; }
  componentDidCatch(error, info) { console.error("Render error:", error, info); }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: 40, gap: 10, minHeight: "60vh" }}>
        <div style={{ fontFamily: DISPLAY, fontSize: 11, letterSpacing: 4, textTransform: "uppercase", color: T.gold }}>Un grimoire s'est dechire</div>
        <h2 style={{ fontFamily: ORNATE, fontSize: 24, color: T.parch, margin: "4px 0" }}>Une erreur est survenue</h2>
        <p style={{ fontFamily: BODY, fontSize: 15, color: T.mistDim, maxWidth: 480, textAlign: "center" }}>
          L'ecran a rencontre un probleme. Tu peux recharger ou revenir a l'accueil.
        </p>
        <code style={{ fontFamily: "monospace", fontSize: 12, color: T.mistDim, maxWidth: 520, overflow: "hidden", textOverflow: "ellipsis" }}>
          {String(this.state.error?.message || this.state.error)}
        </code>
        <div style={{ display: "flex", gap: 10, marginTop: 8 }}>
          <button onClick={() => this.setState({ error: null })} style={btn}>Reessayer</button>
          <button onClick={() => { this.setState({ error: null }); this.props.onHome?.(); }} style={btn}>Accueil</button>
        </div>
      </div>
    );
  }
}
const btn = { padding: "9px 18px", borderRadius: 3, background: "transparent", border: `1px solid ${T.gold}`, color: T.gold,
  cursor: "pointer", fontFamily: DISPLAY, fontSize: 10.5, letterSpacing: 1.5, textTransform: "uppercase" };
