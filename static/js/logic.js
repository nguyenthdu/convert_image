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
  const translationGroup = document.getElementById("translationGroup");
  if (this.checked) {
    translationGroup.style.display = "block";
  }else{
    translationGroup.style.display = "none";
  }
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

    // Gom các file theo từng subfolder.
    // Sử dụng thuộc tính webkitRelativePath để tách tên folder:
    // Ví dụ: "ParentFolder/Subfolder1/File.jpg" → ParentFolder là tên folder cha, Subfolder1 là tên sheet.
    const folderMap = {};
    for (let file of files) {
      // Kiểm tra định dạng file: chỉ xử lý file có type bắt đầu bằng "image/"
      if (!file.type || !file.type.startsWith("image/")) continue;
      const pathParts = file.webkitRelativePath.split("/");
      // Tên folder cha là phần tử đầu tiên
      const parentFolder = pathParts[0];
      // Tên subfolder: nếu có nhiều phần, chúng ta dùng phần giữa (từ index 1 đến trước file)
      const subFolder = pathParts.length > 2 ? pathParts.slice(1, -1).join("/") : "";
      // Dùng subFolder làm key; nếu subFolder rỗng, đặt là "RootFolder"
      const key = subFolder ? subFolder : "RootFolder";
      if (!folderMap[key]) {
        folderMap[key] = [];
      }
      folderMap[key].push(file);
    }

    const subFolderKeys = Object.keys(folderMap);
    if (subFolderKeys.length === 0) {
      errorElement.textContent = "Không tìm thấy file hình ảnh hợp lệ trong thư mục.";
      return;
    }

    // Lấy tên folder cha từ file đầu tiên
    let parentFolderName = files[0].webkitRelativePath.split("/")[0];

    // Tạo 1 workbook
    const workbook = new ExcelJS.Workbook();

    // Xử lý từng subfolder → tạo một sheet cho mỗi subfolder
    for (const key of subFolderKeys) {
      const imageFiles = folderMap[key];
      if (!imageFiles.length) continue;

      // Sử dụng key làm tên sheet, nếu quá 31 ký tự (giới hạn của Excel) thì cắt bớt
      let sheetName = key;
      if (sheetName.length > 31) {
        sheetName = sheetName.substring(0, 31);
      }
      const worksheet = workbook.addWorksheet(sheetName);
      // Cấu hình cột của sheet
      worksheet.columns = [
        { header: "Tên hình", key: "name", width: 30 },
        { header: "Hình ảnh", key: "image", width: 30 },
        { header: "Nội dung hình", key: "text", width: 40 },
        { header: "Dịch", key: "translation", width: 50 }
      ];

      let maxImageWidth = 0;
      for (let i = 0; i < imageFiles.length; i++) {
        const file = imageFiles[i];
        // Cập nhật progress theo từng file trong sheet
        const percent = Math.round(((i + 1) / imageFiles.length) * 100);
        updateProgress(
          percent,
          `Sheet "${sheetName}": Đang xử lý hình ảnh ${i + 1}/${imageFiles.length}`
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

          // Thêm hàng vào sheet với các trường: name, text (nội dung hình) và translation (nội dung dịch)
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
    }

    // Sau khi xử lý hết các sheet, xuất ra 1 file Excel với tên là tên folder cha
    updateProgress(100, "Đang xuất Excel...");
    const buffer = await workbook.xlsx.writeBuffer();
    const blob = new Blob([buffer], {
      type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${parentFolderName}.xlsx`;
    a.click();
    URL.revokeObjectURL(url);

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
    
    // Kiểm tra tên phiên bản đã tồn tại trên sheet đầu tiên (giả sử các sheet đều có cùng header)
    const firstSheet = workbook.worksheets[0];
    const headers = [];
    firstSheet.getRow(1).eachCell((cell) => {
      headers.push(cell.value);
    });
    if (headers.includes(versionName)) {
      errorElement.textContent = `Tên phiên bản "${versionName}" đã tồn tại.`;
      return;
    }

    // Thêm cột phiên bản mới cho mỗi sheet
    workbook.worksheets.forEach(sheet => {
      let lastColumn = 1;
      sheet.getRow(1).eachCell((cell) => {
        lastColumn = cell.col;
      });
      const newColumnIndex = Math.max(5, lastColumn + 1);
      sheet.getCell(1, newColumnIndex).value = versionName;
      sheet.getColumn(newColumnIndex).width = 30;
      // Lưu lại chỉ số cột mới vào thuộc tính của sheet để dùng sau
      sheet.newColumnIndex = newColumnIndex;
    });

    // Tạo map các file từ newImagesFolder, phân loại theo folder con.
    // Giả sử thuộc tính file.webkitRelativePath có dạng "ParentFolder/Subfolder/filename.jpg"
    // Nếu không có subfolder thì đặt key là "RootFolder"
    let imagesByFolder = {};
    Array.from(newImagesFolder).forEach(file => {
      // Kiểm tra định dạng file: chỉ xử lý file có type hợp lệ
      if (!file.type || !file.type.startsWith("image/")) return;
      const parts = file.webkitRelativePath.split("/");
      const folderName = parts.length > 1 ? parts[1] : "RootFolder";
      if (!imagesByFolder[folderName]) {
        imagesByFolder[folderName] = [];
      }
      imagesByFolder[folderName].push(file);
    });
    // Nếu không tìm thấy bất kỳ folder nào, gán tất cả file vào "RootFolder"
    if (Object.keys(imagesByFolder).length === 0) {
      imagesByFolder["RootFolder"] = Array.from(newImagesFolder).filter(file => file.type && file.type.startsWith("image/"));
    }
    // Xử lý từng sheet trong workbook
    for (const sheet of workbook.worksheets) {
      const sheetName = sheet.name;
      // Tìm các file có folder con trùng với tên sheet; nếu không có, sử dụng "RootFolder"
      let matchingFiles = imagesByFolder[sheetName] || imagesByFolder["RootFolder"] || [];
      
      // Tạo map từ tên file đến file đối với matchingFiles
      let newImagesMap = {};
      matchingFiles.forEach(file => {
        newImagesMap[file.name] = file;
      });

      let processedCount = 0;
      const totalRows = sheet.rowCount;

      for (let rowNumber = 2; rowNumber <= totalRows; rowNumber++) {
        const imageName = sheet.getCell(rowNumber, 1).value;
        if (imageName && newImagesMap[imageName]) {
          const newImageFile = newImagesMap[imageName];
          const [dimensionsData, base64Data] = await Promise.all([
            getImageDimensions(newImageFile),
            toBase64(newImageFile),
          ]);
          const dimensions = calculateImageDimensions(dimensionsData.width, dimensionsData.height);
          const imageId = workbook.addImage({
            base64: base64Data,
            extension: newImageFile.name.split(".").pop(),
          });
          sheet.getRow(rowNumber).height = dimensions.height + 5;
          sheet.addImage(imageId, {
            tl: { col: sheet.newColumnIndex - 1, row: rowNumber - 1 },
            ext: dimensions,
            editAs: "oneCell",
          });
        }
        processedCount++;
        updateVersionProgress(
          Math.round((processedCount / totalRows) * 100),
          `Sheet "${sheetName}": Đang xử lý ${processedCount} trên ${totalRows}...`
        );
      }
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
      document.getElementById("updateProgressContainer").style.display = "none";
      updateVersionProgress(0, "");
      location.reload();
    }, 2000);
  } catch (error) {
    console.error("Update error:", error);
    errorElement.textContent = `Update failed: ${error.message}`;
    document.getElementById("updateProgressContainer").style.display = "none";
  }
});
