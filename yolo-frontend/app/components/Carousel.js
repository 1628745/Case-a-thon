"use client";

import { useState } from "react";

export default function Carousel({ slides }) {
  const [current, setCurrent] = useState(0);

  const prev = () => setCurrent((current - 1 + slides.length) % slides.length);
  const next = () => setCurrent((current + 1) % slides.length);

  return (
    <div className="relative w-full max-w-6xl mx-auto my-8">
      <div className="overflow-hidden rounded-xl shadow-lg">
        <img src={slides[current]} alt={`Slide ${current}`} className="w-full h-156 object-cover" />
      </div>
      <button onClick={prev} className="absolute top-1/2 left-2 -translate-y-1/2 bg-white rounded-full p-2 shadow-lg hover:bg-gray-200">◀</button>
      <button onClick={next} className="absolute top-1/2 right-2 -translate-y-1/2 bg-white rounded-full p-2 shadow-lg hover:bg-gray-200">▶</button>
    </div>
  );
}