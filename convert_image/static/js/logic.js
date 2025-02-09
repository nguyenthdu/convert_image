// ==================== SWITCH TAB ====================
function switchTab(tabName) {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.remove("active");
  });
  document.querySelectorAll(".tab-content").forEach((content) => {
    content.classList.remove("active");
  });
  document
    .querySelector(`button[onclick="switchTab('${tabName}')"]`)
    .classList.add("active");
  document.getElementById(`${tabName}Tab`).classList.add("active");
}

// ==================== OPENCV PREPROCESSING ====================
async function preprocessImage(file) {
  return new Promise((resolve, reject) => {
    console.log("Bắt đầu xử lý hình ảnh với OpenCV.js...");
    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");

      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      // Đọc hình ảnh từ canvas
      const src = cv.imread(canvas);
      const gray = new cv.Mat();

      // Chuyển sang grayscale
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);

      // Tăng cường độ tương phản bằng CLAHE
      const clahe = new cv.createCLAHE(2.0, new cv.Size(8, 8));
      clahe.apply(gray, gray);

      // Chuyển đổi ảnh sang đen trắng với adaptive thresholding
      const binary = new cv.Mat();
      cv.adaptiveThreshold(
        gray,
        binary,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        11,
        2
      );

      // Giảm nhiễu với Gaussian Blur
      const ksize = new cv.Size(3, 3);
      cv.GaussianBlur(binary, binary, ksize, 0, 0, cv.BORDER_DEFAULT);

      cv.imshow(canvas, binary);
      canvas.toBlob((blob) => resolve(blob));

      src.delete();
      gray.delete();
      binary.delete();
      URL.revokeObjectURL(url);
    };

    img.onerror = (error) => {
      reject(error);
      URL.revokeObjectURL(url);
    };
    img.src = url;
  });
}

// ==================== OCR & Translation via API ====================
async function recognizeTextViaAPI(file) {
  const formData = new FormData();
  formData.append('image', file);

  // Lấy ngôn ngữ OCR (nguồn) từ giao diện
  const language = document.getElementById("ocrLanguage").value;
  formData.append('language', language);

  // Nếu chức năng dịch được bật, lấy thêm ngôn ngữ đích
  const enableTranslation = document.getElementById("enableTranslation").checked;
  if (enableTranslation) {
    const targetLanguage = document.getElementById("translationLanguage").value;
    formData.append('target_language', targetLanguage);
  }

  try {
    const response = await fetch('http://localhost:5000/api/ocr', {
      method: 'POST',
      body: formData,
    });
    const result = await response.json();
    if (response.ok) {
      // Trả về đối tượng gồm 2 trường: text (nội dung hình) và translation (nội dung dịch)
         //log
         console.log("Nội dung hình:", result.text);
         console.log("Dịch:", result.translation);
      return {
        text: result.text,
        translation: result.translation
      };
   
    } else {
      console.error("Lỗi từ backend:", result.error);
      return { text: "Không lấy được nội dung", translation: "" };
    }
  } catch (error) {
    console.error("Lỗi khi gọi API:", error);
    return { text: "Không lấy được nội dung", translation: "" };
  }
}

// ==================== UTILITIES ====================
function updateProgress(percent, status) {
  const progress = document.getElementById("progress");
  const statusElement = document.getElementById("status");
  const progressContainer = document.getElementById("progressContainer");

  progressContainer.style.display = "block";
  progress.style.width = `${percent}%`;
  statusElement.textContent = status;
}

function toBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(",")[1]);
    reader.onerror = (error) => reject(error);
    reader.readAsDataURL(file);
  });
}

function getImageDimensions(file) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      resolve({ width: img.width, height: img.height });
      URL.revokeObjectURL(url);
    };

    img.onerror = (error) => {
      reject(error);
      URL.revokeObjectURL(url);
    };
    img.src = url;
  });
}

function calculateImageDimensions(originalWidth, originalHeight, maxWidth = 200, maxHeight = 100) {
  const ratio = Math.min(maxWidth / originalWidth, maxHeight / originalHeight);
  return {
    width: Math.floor(originalWidth * ratio),
    height: Math.floor(originalHeight * ratio),
  };
}

// Hiển thị/ẩn dropdown ngôn ngữ OCR
document.getElementById("enableOCR").addEventListener("change", function () {
  const languageGroup = document.getElementById("languageGroup");
  languageGroup.style.display = this.checked ? "block" : "none";
});

// Hiển thị/ẩn dropdown ngôn ngữ dịch
document.getElementById("enableTranslation").addEventListener("change", function () {
  const translationGroup = document.getElementById("translationLanguageGroup");
  translationGroup.style.display = this.checked ? "block" : "none";
});

// ==================== XUẤT RA EXCEL ====================
document.getElementById("exportBtn").addEventListener("click", async () => {
  const folderInput = document.getElementById("folderInput");
  const errorElement = document.getElementById("error");
  const enableOCR = document.getElementById("enableOCR").checked;
  const files = folderInput.files;

  if (files.length === 0) {
    errorElement.textContent = "Vui lòng chọn một thư mục chứa hình ảnh.";
    return;
  }

  try {
    errorElement.textContent = "";

    // Gom các file theo từng subfolder
    const folderMap = {};
    for (let file of files) {
      if (!file.type.startsWith("image/")) continue;
      const pathParts = file.webkitRelativePath.split("/");
      const subFolderPath = pathParts.slice(0, -1).join("/");
      if (!folderMap[subFolderPath]) {
        folderMap[subFolderPath] = [];
      }
      folderMap[subFolderPath].push(file);
    }

    const subFolderPaths = Object.keys(folderMap);
    if (subFolderPaths.length === 0) {
      errorElement.textContent = "Không tìm thấy file hình ảnh hợp lệ trong thư mục.";
      return;
    }

    // Xuất ra file Excel cho từng subfolder
    for (const subFolderPath of subFolderPaths) {
      const imageFiles = folderMap[subFolderPath];
      if (!imageFiles.length) continue;

      let excelFileName = subFolderPath.replace(/\//g, "_");
      if (!excelFileName) {
        excelFileName = "RootFolder";
      }

      updateProgress(
        Math.round((subFolderPaths.indexOf(subFolderPath) / subFolderPaths.length) * 100),
        `Thư mục "${subFolderPath}": Đang xử lý...`
      );

      // Tạo workbook và worksheet với ExcelJS, cấu hình 4 cột: Tên hình, Hình ảnh, Nội dung hình, Dịch
      const workbook = new ExcelJS.Workbook();
      const worksheet = workbook.addWorksheet("Images");
      worksheet.columns = [
        { header: "Tên hình", key: "name", width: 30 },
        { header: "Hình ảnh", key: "image", width: 30 },
        { header: "Nội dung hình", key: "text", width: 40 },
        { header: "Dịch", key: "translation", width: 50 }
      ];

      let maxImageWidth = 0;
      for (let i = 0; i < imageFiles.length; i++) {
        const file = imageFiles[i];
        const percent = Math.round(((i + 1) / imageFiles.length) * 100);
        updateProgress(
          percent,
          `Thư mục "${subFolderPath}": Đang xử lý hình ảnh ${i + 1}/${imageFiles.length}`
        );

        try {
          // Nếu OCR được bật thì gọi API OCR & Dịch, ngược lại trả về đối tượng rỗng
          const recognitionPromise = enableOCR
            ? recognizeTextViaAPI(file)
            : Promise.resolve({ text: "", translation: "" });

          const [dimensionsData, base64Data, ocrResult] = await Promise.all([
            getImageDimensions(file),
            toBase64(file),
            recognitionPromise,
          ]);

          const dimensions = calculateImageDimensions(dimensionsData.width, dimensionsData.height);

          if (dimensions.width > maxImageWidth) {
            maxImageWidth = dimensions.width;
          }

          const imageId = workbook.addImage({
            base64: base64Data,
            extension: file.name.split(".").pop(),
          });

          // Thêm hàng vào worksheet với các trường: name, text (nội dung hình) và translation (nội dung dịch)
          const rowIndex = worksheet.addRow({
            name: file.name,
            text: ocrResult.text,
            translation: ocrResult.translation
          }).number;

          worksheet.getRow(rowIndex).height = dimensions.height + 5;
          worksheet.addImage(imageId, {
            tl: { col: 1, row: rowIndex - 1 },
            ext: dimensions,
            editAs: "oneCell",
          });
        } catch (error) {
          console.error(`Lỗi xử lý hình ảnh ${file.name}:`, error);
        }
      }
      worksheet.getColumn(2).width = maxImageWidth / 7;

      updateProgress(100, `Thư mục "${subFolderPath}": Đang xuất Excel...`);
      const buffer = await workbook.xlsx.writeBuffer();
      const blob = new Blob([buffer], {
        type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${excelFileName}.xlsx`;
      a.click();
      URL.revokeObjectURL(url);
    }

    updateProgress(100, "Xuất thành công!");
    setTimeout(() => {
      document.getElementById("progressContainer").style.display = "none";
      updateProgress(0, "");
      location.reload();
    }, 2000);
  } catch (error) {
    console.error("Xuất lỗi:", error);
    errorElement.textContent = `Xuất lỗi: ${error.message}`;
    document.getElementById("progressContainer").style.display = "none";
  }
});

// ==================== CẬP NHẬT PHIÊN BẢN ====================
function updateVersionProgress(percent, status) {
  const progress = document.getElementById("updateProgress");
  const statusElement = document.getElementById("updateStatus");
  const progressContainer = document.getElementById("updateProgressContainer");
  progressContainer.style.display = "block";
  progress.style.width = `${percent}%`;
  statusElement.textContent = status;
}

document.getElementById("updateBtn").addEventListener("click", async () => {
  const excelFile = document.getElementById("excelFile").files[0];
  const versionName = document.getElementById("versionName").value.trim();
  const newImagesFolder = document.getElementById("newImagesFolder").files;
  const errorElement = document.getElementById("updateError");

  if (!excelFile) {
    errorElement.textContent = "Vui lòng chọn file Excel.";
    return;
  }
  if (!versionName) {
    errorElement.textContent = "Vui lòng nhập tên phiên bản.";
    return;
  }
  if (newImagesFolder.length === 0) {
    errorElement.textContent = "Vui lòng chọn thư mục chứa hình ảnh.";
    return;
  }

  try {
    errorElement.textContent = "";
    updateVersionProgress(0, "Đang xử lý...");

    const workbook = new ExcelJS.Workbook();
    const excelData = await excelFile.arrayBuffer();
    await workbook.xlsx.load(excelData);
    const worksheet = workbook.worksheets[0];

    // Kiểm tra tên phiên bản đã tồn tại chưa
    const headers = [];
    worksheet.getRow(1).eachCell((cell) => {
      headers.push(cell.value);
    });
    if (headers.includes(versionName)) {
      errorElement.textContent = `Tên phiên bản "${versionName}" đã tồn tại.`;
      return;
    }

    // Xác định cột mới
    let lastColumn = 1;
    worksheet.getRow(1).eachCell((cell) => {
      lastColumn = cell.col;
    });
    const newColumnIndex = Math.max(5, lastColumn + 1);
    worksheet.getCell(1, newColumnIndex).value = versionName;
    worksheet.getColumn(newColumnIndex).width = 30;

    let processedCount = 0;
    const totalRows = worksheet.rowCount;

    for (let rowNumber = 2; rowNumber <= totalRows; rowNumber++) {
      const imageName = worksheet.getCell(rowNumber, 1).value;
      if (imageName && newImagesFolder && newImagesFolder.length > 0) {
        // Tạo map cho hình ảnh mới
        let newImages = {};
        Array.from(newImagesFolder).forEach((file) => {
          if (file.type.startsWith("image/")) {
            newImages[file.name] = file;
          }
        });
        if (newImages[imageName]) {
          const newImageFile = newImages[imageName];
          const [originalDimensions, base64Data] = await Promise.all([
            getImageDimensions(newImageFile),
            toBase64(newImageFile),
          ]);
          const dimensions = calculateImageDimensions(
            originalDimensions.width,
            originalDimensions.height
          );

          const imageId = workbook.addImage({
            base64: base64Data,
            extension: newImageFile.name.split(".").pop(),
          });
          worksheet.getRow(rowNumber).height = dimensions.height + 5;
          worksheet.addImage(imageId, {
            tl: {
              col: newColumnIndex - 1,
              row: rowNumber - 1,
            },
            ext: dimensions,
            editAs: "oneCell",
          });
        }
      }
      processedCount++;
      updateVersionProgress(
        Math.round((processedCount / totalRows) * 100),
        `Đang xử lý ${processedCount} trên ${totalRows}...`
      );
    }

    updateVersionProgress(90, "Đang tải xuống...");
    const buffer = await workbook.xlsx.writeBuffer();
    const blob = new Blob([buffer], {
      type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    let filename = excelFile.name.replace(".xlsx", "");
    a.download = `${filename}_${versionName}.xlsx`;
    a.click();
    URL.revokeObjectURL(url);

    updateVersionProgress(100, "Thêm phiên bản thành công!");
    setTimeout(() => {
      document.getElementById("updateProgressContainer").style.display =
        "none";
      updateVersionProgress(0, "");
      location.reload();
    }, 2000);
  } catch (error) {
    console.error("Update error:", error);
    errorElement.textContent = `Update failed: ${error.message}`;
    document.getElementById("updateProgressContainer").style.display = "none";
  }
});
