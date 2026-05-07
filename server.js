const express = require("express");
const cors = require("cors");
const { spawn } = require("child_process");

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors({ origin: "*" }));
app.use(express.json());

const FIREBASE_URL =
  "https://solar-ml-system-default-rtdb.firebaseio.com/solar_readings.json";

app.get("/", (req, res) => {
  res.json({
    message: "Solar ML backend is running",
    status: "online",
  });
});

app.get("/api/power", async (req, res) => {
  try {
    const response = await fetch(FIREBASE_URL);

    if (!response.ok) {
      return res.status(500).json({
        error: "Failed to fetch Firebase data",
      });
    }

    const firebaseData = await response.json();

    if (!firebaseData) {
      return res.json([]);
    }

    const rows = Object.values(firebaseData)
      .filter(
        (item) =>
          item &&
          typeof item === "object" &&
          item.timestamp &&
          String(item.timestamp).includes("T")
      )
      .map((item) => {
        const timestamp = String(item.timestamp);

        return {
          timestamp,
          time: timestamp.substring(11, 16),
          power: Number(item.power || 0),
          solar: Number(item.power || 0),
          battery: Number(item.battery || 0),
          irradiance: Number(item.irradiance || 0),
          temperature: Number(item.temperature || 0),
          humidity: Number(item.humidity || 0),
          voltage: Number(item.voltage || 0),
          current: Number(item.current || 0),
        };
      })
      .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

    res.json(rows);
  } catch (error) {
    console.error("Error in /api/power:", error);
    res.status(500).json({
      error: "Failed to load power data",
    });
  }
});

app.get("/api/predict", (req, res) => {
  const python = spawn("python", ["predict_latest.py"], {
    cwd: __dirname,
  });

  let output = "";
  let errorOutput = "";

  python.stdout.on("data", (data) => {
    output += data.toString();
  });

  python.stderr.on("data", (data) => {
    errorOutput += data.toString();
  });

  python.on("close", (code) => {
    if (code === 0) {
      try {
        const result = JSON.parse(output);
        res.json(result);
      } catch (err) {
        res.status(500).json({
          success: false,
          error: "Failed to parse prediction output",
          raw: output,
        });
      }
    } else {
      res.status(500).json({
        success: false,
        error: errorOutput,
      });
    }
  });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
