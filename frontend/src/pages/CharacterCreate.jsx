/** Creation d'un personnage hors campagne (accessible a tous). */
import { ScreenTitle, Divider } from "../components/ornaments.jsx";
import { CharacterForm } from "../components/CharacterForm.jsx";
import { createCharacter } from "../api/characters.js";

export function CharacterCreate({ setView }) {
  const save = async (payload) => {
    await createCharacter(payload);
    setView("library");           // -> on retrouve la fiche dans Mon antre
  };
  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", minHeight: 0 }}>
      <ScreenTitle eyebrow="Forge de heros" title="Creer un personnage" />
      <div style={{ padding: "0 48px" }}><Divider label="Fiche de personnage" /></div>
      <div style={{ flex: 1, overflowY: "auto", padding: "8px 48px 32px" }}>
        <CharacterForm onSave={save} submitLabel="Enregistrer le personnage" />
      </div>
    </div>
  );
}
