/** Suit la largeur de la fenetre pour adapter la mise en page (responsive). */
import { useEffect, useState } from "react";

export function useViewport() {
  const [width, setWidth] = useState(typeof window !== "undefined" ? window.innerWidth : 1200);
  useEffect(() => {
    const onResize = () => setWidth(window.innerWidth);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  return { width, isMobile: width < 760, isNarrow: width < 1040 };
}
