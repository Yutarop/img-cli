// image_mover.go
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"
)

type ClusterData struct {
	ImagePaths    []string `json:"image_paths"`
	ClusterLabels []int    `json:"cluster_labels"`
	OutputFolder  string   `json:"output_folder"`
	NClusters     int      `json:"n_clusters"`
}

type MoveTask struct {
	SourcePath      string
	DestinationPath string
	ClusterID       int
}

type MoveResult struct {
	SourcePath      string `json:"source_path"`
	DestinationPath string `json:"destination_path"`
	ClusterID       int    `json:"cluster_id"`
	Success         bool   `json:"success"`
	Error           string `json:"error,omitempty"`
}

type ProcessResult struct {
	MovedCount            int      `json:"moved_count"`
	ProcessingTimeSeconds float64  `json:"processing_time_seconds"`
	TotalTasks            int      `json:"total_tasks"`
	SuccessfulMoves       int      `json:"successful_moves"`
	Errors                []string `json:"errors"`
}

type ImageMover struct {
	numWorkers int
}

func NewImageMover(numWorkers int) *ImageMover {
	return &ImageMover{
		numWorkers: numWorkers,
	}
}

func (im *ImageMover) copyFile(src, dst string) error {
	// 元ファイルを開く
	sourceFile, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("failed to open source file: %w", err)
	}
	defer sourceFile.Close()

	// コピー先ディレクトリを作成
	if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
		return fmt.Errorf("failed to create destination directory: %w", err)
	}

	// コピー先ファイルを作成
	destinationFile, err := os.Create(dst)
	if err != nil {
		return fmt.Errorf("failed to create destination file: %w", err)
	}
	defer destinationFile.Close()

	// ファイルをコピー
	_, err = io.Copy(destinationFile, sourceFile)
	if err != nil {
		return fmt.Errorf("failed to copy file content: %w", err)
	}

	return nil
}

func (im *ImageMover) moveImagesParallel(clusterData *ClusterData) (*ProcessResult, error) {
	startTime := time.Now()

	// 移動タスクを準備
	var tasks []MoveTask
	for i, imagePath := range clusterData.ImagePaths {
		if i >= len(clusterData.ClusterLabels) {
			continue
		}

		clusterID := clusterData.ClusterLabels[i]
		filename := filepath.Base(imagePath)
		clusterFolder := fmt.Sprintf("cluster_%d", clusterID)
		destinationPath := filepath.Join(clusterData.OutputFolder, clusterFolder, filename)

		tasks = append(tasks, MoveTask{
			SourcePath:      imagePath,
			DestinationPath: destinationPath,
			ClusterID:       clusterID,
		})
	}

	if len(tasks) == 0 {
		return &ProcessResult{
			MovedCount:            0,
			ProcessingTimeSeconds: time.Since(startTime).Seconds(),
			TotalTasks:            0,
			SuccessfulMoves:       0,
			Errors:                []string{"No tasks to process"},
		}, nil
	}

	// クラスタフォルダを事前に作成
	for i := 0; i < clusterData.NClusters; i++ {
		clusterFolder := filepath.Join(clusterData.OutputFolder, fmt.Sprintf("cluster_%d", i))
		if err := os.MkdirAll(clusterFolder, 0755); err != nil {
			return nil, fmt.Errorf("failed to create cluster folder %s: %w", clusterFolder, err)
		}
	}

	// ワーカープールパターンで並列処理
	taskChan := make(chan MoveTask, len(tasks))
	resultChan := make(chan MoveResult, len(tasks))
	var wg sync.WaitGroup

	// ワーカーを起動
	for i := 0; i < im.numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for task := range taskChan {
				err := im.copyFile(task.SourcePath, task.DestinationPath)
				result := MoveResult{
					SourcePath:      task.SourcePath,
					DestinationPath: task.DestinationPath,
					ClusterID:       task.ClusterID,
					Success:         err == nil,
				}
				if err != nil {
					result.Error = err.Error()
				}
				resultChan <- result
			}
		}(i)
	}

	// タスクをチャネルに送信
	go func() {
		defer close(taskChan)
		for _, task := range tasks {
			taskChan <- task
		}
	}()

	// ワーカーの完了を待つ
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// 結果を収集
	var errors []string
	successCount := 0
	totalCount := 0

	for result := range resultChan {
		totalCount++
		if result.Success {
			successCount++
		} else {
			errors = append(errors, fmt.Sprintf("%s -> %s: %s",
				result.SourcePath, result.DestinationPath, result.Error))
		}
	}

	processingTime := time.Since(startTime).Seconds()

	return &ProcessResult{
		MovedCount:            successCount,
		ProcessingTimeSeconds: processingTime,
		TotalTasks:            totalCount,
		SuccessfulMoves:       successCount,
		Errors:                errors,
	}, nil
}

func loadClusterData(filename string) (*ClusterData, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open cluster data file: %w", err)
	}
	defer file.Close()

	var clusterData ClusterData
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&clusterData); err != nil {
		return nil, fmt.Errorf("failed to decode cluster data: %w", err)
	}

	return &clusterData, nil
}

func main() {
	var dataFile = flag.String("data", "", "Path to cluster data JSON file")
	var numWorkers = flag.Int("workers", runtime.NumCPU(), "Number of worker goroutines")
	flag.Parse()

	if *dataFile == "" {
		log.Fatal("Cluster data file must be specified")
	}

	// クラスタデータを読み込み
	clusterData, err := loadClusterData(*dataFile)
	if err != nil {
		log.Fatalf("Failed to load cluster data: %v", err)
	}

	mover := NewImageMover(*numWorkers)

	fmt.Fprintf(os.Stderr, "Starting parallel image moving with %d workers...\n", *numWorkers)
	fmt.Fprintf(os.Stderr, "Total images to move: %d\n", len(clusterData.ImagePaths))
	fmt.Fprintf(os.Stderr, "Output folder: %s\n", clusterData.OutputFolder)
	fmt.Fprintf(os.Stderr, "Number of clusters: %d\n", clusterData.NClusters)

	result, err := mover.moveImagesParallel(clusterData)
	if err != nil {
		log.Fatalf("Moving failed: %v", err)
	}

	// 結果をJSON形式で標準出力に出力
	jsonResult, err := json.Marshal(result)
	if err != nil {
		log.Fatalf("Failed to marshal result to JSON: %v", err)
	}

	fmt.Println(string(jsonResult))

	// ログ情報を標準エラーに出力
	fmt.Fprintf(os.Stderr, "Moving completed successfully\n")
	fmt.Fprintf(os.Stderr, "Total tasks: %d, Successful: %d, Errors: %d\n",
		result.TotalTasks, result.SuccessfulMoves, len(result.Errors))
	fmt.Fprintf(os.Stderr, "Moving time: %.2f seconds\n", result.ProcessingTimeSeconds)
}
