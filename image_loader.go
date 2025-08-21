// image_loader.go
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/nfnt/resize"
)

type ImageInfo struct {
	OriginalPath  string `json:"original_path"`
	ProcessedPath string `json:"processed_path"`
	Width         int    `json:"width"`
	Height        int    `json:"height"`
	Error         string `json:"error,omitempty"`
}

type ProcessResult struct {
	ImagePaths            []string    `json:"image_paths"`
	LoadedImages          []ImageInfo `json:"loaded_images"`
	ProcessingTimeSeconds float64     `json:"processing_time_seconds"`
	TotalImages           int         `json:"total_images"`
	SuccessfulImages      int         `json:"successful_images"`
	Errors                []string    `json:"errors"`
}

type ImageProcessor struct {
	inputDir      string
	outputDir     string
	numWorkers    int
	targetSize    int
	supportedExts map[string]bool
}

func NewImageProcessor(inputDir, outputDir string, numWorkers, targetSize int) *ImageProcessor {
	supportedExts := map[string]bool{
		".jpg":  true,
		".jpeg": true,
		".png":  true,
		".bmp":  true,
		".tiff": true,
		".tif":  true,
	}

	return &ImageProcessor{
		inputDir:      inputDir,
		outputDir:     outputDir,
		numWorkers:    numWorkers,
		targetSize:    targetSize,
		supportedExts: supportedExts,
	}
}

func (ip *ImageProcessor) isValidImageFile(filename string) bool {
	ext := strings.ToLower(filepath.Ext(filename))
	return ip.supportedExts[ext]
}

func (ip *ImageProcessor) processImage(imagePath string, outputPath string) (*ImageInfo, error) {
	// ファイルの存在確認
	if _, err := os.Stat(imagePath); os.IsNotExist(err) {
		return nil, fmt.Errorf("file does not exist: %s", imagePath)
	}

	// 画像ファイルを開く
	file, err := os.Open(imagePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file %s: %w", imagePath, err)
	}
	defer file.Close()

	// 画像形式を判定してデコード
	var img image.Image
	ext := strings.ToLower(filepath.Ext(imagePath))

	switch ext {
	case ".jpg", ".jpeg":
		img, err = jpeg.Decode(file)
	case ".png":
		img, err = png.Decode(file)
	default:
		// その他の形式は汎用デコーダーを使用
		file.Seek(0, 0) // ファイルポインタを先頭に戻す
		img, _, err = image.Decode(file)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to decode image %s: %w", imagePath, err)
	}

	if img == nil {
		return nil, fmt.Errorf("decoded image is nil: %s", imagePath)
	}

	// 画像をリサイズ（224x224、ResNet50の入力サイズ）
	resizedImg := resize.Resize(uint(ip.targetSize), uint(ip.targetSize), img, resize.Lanczos3)
	if resizedImg == nil {
		return nil, fmt.Errorf("failed to resize image: %s", imagePath)
	}

	// 画像データを正規化してnumpy配列形式で保存
	bounds := resizedImg.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	// RGBの値を取得して配列に変換
	data := make([]float64, height*width*3)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := resizedImg.At(x, y).RGBA()
			// 0-255の範囲に正規化（RGBA値は0-65535の範囲なので8ビット右シフト）
			data[y*width*3+x*3+0] = float64(r >> 8) // R
			data[y*width*3+x*3+1] = float64(g >> 8) // G
			data[y*width*3+x*3+2] = float64(b >> 8) // B
		}
	}

	// 出力ディレクトリを作成
	if err := os.MkdirAll(outputPath, 0755); err != nil {
		return nil, fmt.Errorf("failed to create output directory: %w", err)
	}

	// バイナリファイルとして保存（Pythonで読み込み可能）
	baseName := strings.TrimSuffix(filepath.Base(imagePath), filepath.Ext(imagePath))
	processedFile := filepath.Join(outputPath, baseName+".bin")

	if err := saveBinaryArray(processedFile, data); err != nil {
		return nil, fmt.Errorf("failed to save processed image %s: %w", processedFile, err)
	}

	return &ImageInfo{
		OriginalPath:  imagePath,
		ProcessedPath: processedFile,
		Width:         width,
		Height:        height,
	}, nil
}

// バイナリファイルとして保存
func saveBinaryArray(filename string, data []float64) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", filename, err)
	}
	defer file.Close()

	// Little endianでfloat64を書き込み
	for _, val := range data {
		if err := binary.Write(file, binary.LittleEndian, val); err != nil {
			return fmt.Errorf("failed to write binary data: %w", err)
		}
	}

	return nil
}

func (ip *ImageProcessor) processImagesParallel() (*ProcessResult, error) {
	startTime := time.Now()

	// 入力ディレクトリから画像ファイルを収集
	var imagePaths []string
	err := filepath.Walk(ip.inputDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: Error walking path %s: %v\n", path, err)
			return nil // エラーがあっても処理を続行
		}
		if !info.IsDir() && ip.isValidImageFile(info.Name()) {
			imagePaths = append(imagePaths, path)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("failed to collect image files: %w", err)
	}

	fmt.Fprintf(os.Stderr, "Found %d image files\n", len(imagePaths))

	if len(imagePaths) == 0 {
		return &ProcessResult{
			ImagePaths:            []string{},
			LoadedImages:          []ImageInfo{},
			ProcessingTimeSeconds: time.Since(startTime).Seconds(),
			TotalImages:           0,
			SuccessfulImages:      0,
			Errors:                []string{"No supported image files found in " + ip.inputDir},
		}, nil
	}

	// 出力ディレクトリを作成
	if err := os.MkdirAll(ip.outputDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create output directory: %w", err)
	}

	// ワーカープールパターンを使用した並列処理
	imageChan := make(chan string, len(imagePaths))
	resultChan := make(chan ImageInfo, len(imagePaths))
	var wg sync.WaitGroup

	// ワーカーを起動
	for i := 0; i < ip.numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for imagePath := range imageChan {
				fmt.Fprintf(os.Stderr, "Worker %d processing: %s\n", workerID, imagePath)
				result, err := ip.processImage(imagePath, ip.outputDir)
				if err != nil {
					fmt.Fprintf(os.Stderr, "Worker %d error processing %s: %v\n", workerID, imagePath, err)
					resultChan <- ImageInfo{
						OriginalPath: imagePath,
						Error:        err.Error(),
					}
				} else {
					fmt.Fprintf(os.Stderr, "Worker %d successfully processed: %s\n", workerID, imagePath)
					resultChan <- *result
				}
			}
		}(i)
	}

	// 画像パスをチャンネルに送信
	go func() {
		defer close(imageChan)
		for _, imagePath := range imagePaths {
			imageChan <- imagePath
		}
	}()

	// ワーカーの完了を待つ
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// 結果を収集
	var loadedImages []ImageInfo
	var errors []string
	successCount := 0

	for result := range resultChan {
		if result.Error != "" {
			errors = append(errors, fmt.Sprintf("%s: %s", result.OriginalPath, result.Error))
		} else {
			loadedImages = append(loadedImages, result)
			successCount++
		}
	}

	processingTime := time.Since(startTime).Seconds()

	return &ProcessResult{
		ImagePaths:            imagePaths,
		LoadedImages:          loadedImages,
		ProcessingTimeSeconds: processingTime,
		TotalImages:           len(imagePaths),
		SuccessfulImages:      successCount,
		Errors:                errors,
	}, nil
}

func main() {
	var inputDir = flag.String("input", "", "Input directory containing images")
	var outputDir = flag.String("output", "", "Output directory for processed images")
	var numWorkers = flag.Int("workers", runtime.NumCPU(), "Number of worker goroutines")
	var targetSize = flag.Int("size", 224, "Target image size (default: 224 for ResNet50)")
	flag.Parse()

	if *inputDir == "" || *outputDir == "" {
		log.Fatal("Both input and output directories must be specified")
	}

	// 入力ディレクトリの存在確認
	if _, err := os.Stat(*inputDir); os.IsNotExist(err) {
		log.Fatalf("Input directory does not exist: %s", *inputDir)
	}

	processor := NewImageProcessor(*inputDir, *outputDir, *numWorkers, *targetSize)

	fmt.Fprintf(os.Stderr, "Starting parallel image processing with %d workers...\n", *numWorkers)
	fmt.Fprintf(os.Stderr, "Input directory: %s\n", *inputDir)
	fmt.Fprintf(os.Stderr, "Output directory: %s\n", *outputDir)

	result, err := processor.processImagesParallel()
	if err != nil {
		log.Fatalf("Processing failed: %v", err)
	}

	// 結果をJSON形式で標準出力に出力
	jsonResult, err := json.Marshal(result)
	if err != nil {
		log.Fatalf("Failed to marshal result to JSON: %v", err)
	}

	fmt.Println(string(jsonResult))

	// ログ情報を標準エラーに出力
	fmt.Fprintf(os.Stderr, "Processing completed successfully\n")
	fmt.Fprintf(os.Stderr, "Total images: %d, Successful: %d, Errors: %d\n",
		result.TotalImages, result.SuccessfulImages, len(result.Errors))
	fmt.Fprintf(os.Stderr, "Processing time: %.2f seconds\n", result.ProcessingTimeSeconds)

	if len(result.Errors) > 0 {
		fmt.Fprintf(os.Stderr, "Errors encountered:\n")
		for _, errMsg := range result.Errors {
			fmt.Fprintf(os.Stderr, "  %s\n", errMsg)
		}
	}
}
