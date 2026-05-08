const express = require("express");
const cors = require("cors");
const { spawn } = require("child_process");

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors({ origin: "*" }));
app.use(express.json());

const FIREBASE_BASE =
  "https://solar-ml-system-default-rtdb.firebaseio.com/solar_readings.json";

app.get("/", (req, res) => {
  res.json({
    success: true,
    message: "Solar ML backend is running",
    port: PORT,
    status: "online",
  });
});

app.get("/api/power", async (req, res) => {
  try {
    const firebaseURL =
      `${FIREBASE_BASE}?orderBy="$key"&limitToLast=60`;

    const response = await fetch(firebaseURL);

    if (!response.ok) {
      return res.status(500).json({
        success: false,
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
          solar: Number(item.power || 0),
          power: Number(item.power || 0),
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
    console.error("Power API error:", error);
    res.status(500).json({
      success: false,
      error: String(error),
    });
  }
});

app.get("/api/predict", (req, res) => {
  try {
    const pythonProcess = spawn("python3", ["predict_latest.py"], {
      cwd: __dirname,
    });

    let output = "";
    let errorOutput = "";

    pythonProcess.stdout.on("data", (data) => {
      output += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
      errorOutput += data.toString();
    });

    pythonProcess.on("close", (code) => {
      if (code === 0) {
        try {
          res.json(JSON.parse(output));
        } catch {
          res.status(500).json({
            success: false,
            error: "Prediction output parse failed",
            raw_output: output,
          });
        }
      } else {
        res.status(500).json({
          success: false,
          error: errorOutput,
        });
      }
    });
  } catch (err) {
    res.status(500).json({
      success: false,
      error: String(err),
    });
  }
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Server running on port ${PORT}`);
});
