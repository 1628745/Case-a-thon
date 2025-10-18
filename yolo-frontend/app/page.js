"use client";

import { useState } from "react";
import Carousel from "./components/Carousel";

export default function Home() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [zipFile, setZipFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Single image upload
  const handleTestPhoto = async () => {
    if (!selectedImage) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", selectedImage);

    const res = await fetch("https://yolo-backend-dmn3.onrender.com/predict/", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    setResult({ image_base64: data.image_base64, detections: data.detections });
    setLoading(false);
  };

  // Function to group duplicates and average confidence
  const getGroupedDetections = (detections) => {
    const grouped = detections.reduce((acc, det) => {
      const label = det.label;
      if (!acc[label]) {
        acc[label] = { count: 1, confidenceSum: det.confidence };
      } else {
        acc[label].count += 1;
        acc[label].confidenceSum += det.confidence;
      }
      return acc;
    }, {});

    return Object.entries(grouped).map(([label, { count, confidenceSum }]) => ({
      label,
      count,
      avgConfidence: confidenceSum / count,
    }));
  };

  return (
<main className="px-4 min-h-screen" style={{ backgroundColor: "#1e3773" }}>
      {/* Header */}
      <header className="text-center py-8 bg-gradient-to-r from-blue-400 to-purple-500 text-white shadow-md">
        <h1 className="text-4xl font-bold">YOLO Object Detection</h1>
      </header>

      {/* Carousels */}
    <h2 className="text-2xl font-semibold text-white text-center my-6">
    Presentation 1
  </h2>
      <div className="my-8 flex justify-center">
        <Carousel slides={["/img1.jpg","/img2.jpg", "/img3.jpg", "/img4.jpg", "/img5.jpg"]} className="w-3/4" />
      </div>
    <h2 className="text-2xl font-semibold text-white text-center my-6">
    Presentation 2
  </h2>
      <div className="my-8 flex justify-center">
        <Carousel slides={["/imgA.jpg","/imgB.jpg"]} className="w-3/4" />
      </div>
<h2 className="text-2xl font-semibold text-white text-center my-6">
    Our Solution
  </h2>
      {/* Upload Section */}
      <div className="flex justify-center items-start space-x-8 my-8">
        {/* Left: Uploaded Image */}
        <div className="w-1/2 h-160 border-4 border-indigo-300 rounded-xl flex items-center justify-center bg-slate-300">
          {selectedImage ? (
            <img
              src={zipFile ? "/zip.jpg" : URL.createObjectURL(selectedImage)}
              alt="Selected"
              className="max-h-full max-w-full rounded-lg"
            />
          ) : (
            <p className="text-gray-400">No image selected</p>
          )}
        </div>

        {/* Right: Buttons */}
        <div className="w-1/2 h-160 border-4 border-indigo-300 rounded-xl flex flex-col items-center justify-center space-y-4 bg-slate-300">
          <input
            type="file"
            accept="image/*"
            onChange={(e) => { setSelectedImage(e.target.files[0]); setZipFile(null); }}
            className="hidden"
            id="fileInput"
          />
          <label htmlFor="fileInput" className="w-48 py-4 bg-blue-500 text-white font-semibold rounded-xl text-center cursor-pointer hover:bg-blue-600">
            Select Photo
          </label>

          <button
            onClick={handleTestPhoto}
            className="w-48 py-4 bg-blue-500 text-white font-semibold rounded-xl hover:bg-blue-600 transition-all"
          >
            {loading ? "Processing..." : "Test Selected Photo"}
          </button>
        </div>
      </div>

      {/* Results Section */}
      {result && (
        <div className="flex justify-center items-start space-x-8 my-8">
          {/* Left: Annotated Image */}
          <div className="w-1/2 h-160 border-4 border-indigo-300 rounded-xl flex items-center justify-center bg-slate-300">
            {!result.zipMode && result.image_base64 ? (
              <img src={`data:image/jpeg;base64,${result.image_base64}`} alt="Detected" className="max-h-full max-w-full rounded-lg" />
            ) : (
              <img src="/zip.jpg" alt="ZIP" className="max-h-full max-w-full rounded-lg" />
            )}
          </div>

          {/* Right: Text Details */}
          <div className="w-1/2 h-160 border-4 border-indigo-300 rounded-xl p-4 bg-slate-300 overflow-auto">
            {getGroupedDetections(result.detections).map((d, i) => (
              <div key={i} className="text-gray-800 mb-2">
                <p>
                  <span className="font-semibold">Identified animal{d.count > 1 ? "s" : ""}:</span> {d.label}{d.count > 1 ? ` (x${d.count})` : ""}
                </p>
                <p>
                  <span className="font-semibold">Confidence:</span> {(d.avgConfidence * 100).toFixed(1)}%
                </p>
              </div>
            ))}
          </div>
        </div>

      )}
    <h2 className="text-2xl font-semibold text-white text-center my-6">
    Model Statistics
  </h2>
    <img src="/confusion_matrix_normalized.png" alt="Confusion Matrix" className="mx-auto my-8 max-w-full rounded-lg" />
    <img src="/BoxPR_curve.png" alt="Confusion Matrix" className="mx-auto my-8 max-w-full rounded-lg" />


    </main>
  );
}
