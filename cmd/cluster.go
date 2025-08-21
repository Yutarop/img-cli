package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
)

var (
	inputDir    string
	outputDir   string
	clusters    int
	workers     int
	pythonPath  string
	noPCA       bool
	varRatio    float64
	noVisualize bool
	verbose     bool
	force       bool
)

var clusterCmd = &cobra.Command{
	Use:   "cluster [flags]",
	Short: "Cluster images by visual similarity",
	Long: `Cluster images based on visual similarity using ResNet50 features and K-means clustering.
By default, it processes all images in the current directory and saves clustered results to './clustered_images'.

The clustering process involves:
1. Loading images and extracting ResNet50 features
2. Optional PCA dimensionality reduction
3. K-means clustering based on visual similarity  
4. Organizing images into cluster-specific folders
5. Optional visualization generation`,
	Example: `  # Cluster images in current directory
  imgcli cluster

  # Specify input and output directories
  imgcli cluster -i ./photos -o ./clustered_photos

  # Use 3 clusters with 4 workers
  imgcli cluster -k 3 -w 4

  # Disable PCA and visualization
  imgcli cluster --no-pca --no-viz

  # Force overwrite existing output
  imgcli cluster --force`,
	PreRunE: validateClusterFlags,
	RunE:    runCluster,
}

func init() {
	currentDir, _ := os.Getwd()

	clusterCmd.Flags().StringVarP(&inputDir, "input", "i", currentDir, "Input directory containing images")
	clusterCmd.Flags().StringVarP(&outputDir, "output", "o", filepath.Join(currentDir, "clustered_images"), "Output directory for clustered images")
	clusterCmd.Flags().IntVarP(&clusters, "clusters", "k", 5, "Number of clusters (1-50)")
	clusterCmd.Flags().IntVarP(&workers, "workers", "w", 8, "Number of parallel workers (1-32)")
	clusterCmd.Flags().StringVarP(&pythonPath, "python", "p", "python", "Python executable path")
	clusterCmd.Flags().BoolVar(&noPCA, "no-pca", false, "Disable PCA dimensionality reduction")
	clusterCmd.Flags().Float64Var(&varRatio, "variance-ratio", 0.95, "PCA variance ratio to retain (0.5-1.0)")
	clusterCmd.Flags().BoolVar(&noVisualize, "no-viz", false, "Disable cluster visualization")
	clusterCmd.Flags().BoolVar(&force, "force", false, "Force overwrite existing output directory")

	// Bind verbose flag from parent
	clusterCmd.Flags().BoolVarP(&verbose, "verbose", "v", false, "Enable verbose output")
}

func validateClusterFlags(cmd *cobra.Command, args []string) error {
	// Validate cluster count
	if clusters < 1 || clusters > 50 {
		return fmt.Errorf("cluster count must be between 1 and 50, got %d", clusters)
	}

	// Validate worker count
	if workers < 1 || workers > 32 {
		return fmt.Errorf("worker count must be between 1 and 32, got %d", workers)
	}

	// Validate variance ratio
	if varRatio < 0.5 || varRatio > 1.0 {
		return fmt.Errorf("variance ratio must be between 0.5 and 1.0, got %.2f", varRatio)
	}

	// Validate input directory
	if _, err := os.Stat(inputDir); os.IsNotExist(err) {
		return fmt.Errorf("input directory does not exist: %s", inputDir)
	}

	// Check if input directory is readable
	if file, err := os.Open(inputDir); err != nil {
		return fmt.Errorf("cannot read input directory: %s", err)
	} else {
		file.Close()
	}

	// Check output directory conflicts
	if !force {
		if _, err := os.Stat(outputDir); !os.IsNotExist(err) {
			return fmt.Errorf("output directory already exists: %s (use --force to overwrite)", outputDir)
		}
	}

	return nil
}

func runCluster(cmd *cobra.Command, args []string) error {
	if verbose {
		fmt.Printf("Verbose mode enabled\n")
		fmt.Printf("Input directory: %s\n", inputDir)
		fmt.Printf("Output directory: %s\n", outputDir)
		fmt.Printf("Python path: %s\n", pythonPath)
	}

	// Check Python availability first
	if err := checkPythonEnvironment(); err != nil {
		return fmt.Errorf("Python environment check failed: %w", err)
	}

	// Scan for supported image files
	supportedFormats := []string{".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
	imageFiles, err := scanImageFiles(inputDir, supportedFormats)
	if err != nil {
		return fmt.Errorf("failed to scan input directory: %w", err)
	}

	if len(imageFiles) == 0 {
		return fmt.Errorf("no supported image files found in %s\nSupported formats: %s",
			inputDir, strings.Join(supportedFormats, ", "))
	}

	fmt.Printf("Found %d images in %s\n", len(imageFiles), inputDir)

	// Adjust cluster count if necessary
	actualClusters := clusters
	if len(imageFiles) < clusters {
		actualClusters = len(imageFiles)
		fmt.Printf("Adjusted cluster count from %d to %d (limited by image count)\n",
			clusters, actualClusters)
		clusters = actualClusters
	}

	// Create output directory with force option handling
	if err := prepareOutputDirectory(); err != nil {
		return err
	}

	// Display configuration
	printConfiguration(len(imageFiles), actualClusters)

	// Execute clustering pipeline
	return runClusteringPipeline()
}

func checkPythonEnvironment() any {
	panic("unimplemented")
}

func scanImageFiles(dir string, supportedFormats []string) ([]string, error) {
	var imageFiles []string

	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			if verbose {
				fmt.Printf("Warning: Error accessing %s: %v\n", path, err)
			}
			return nil // Continue walking
		}

		if info.IsDir() {
			return nil
		}

		ext := strings.ToLower(filepath.Ext(info.Name()))
		for _, format := range supportedFormats {
			if ext == format {
				imageFiles = append(imageFiles, path)
				break
			}
		}
		return nil
	})

	return imageFiles, err
}

func prepareOutputDirectory() error {
	if force {
		// Remove existing directory if force is enabled
		if err := os.RemoveAll(outputDir); err != nil {
			return fmt.Errorf("failed to remove existing output directory: %w", err)
		}
	}

	// Create output directory
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	return nil
}

func printConfiguration(imageCount, actualClusters int) {
	fmt.Printf("\nConfiguration:\n")
	fmt.Printf("  Input directory: %s\n", inputDir)
	fmt.Printf("  Output directory: %s\n", outputDir)
	fmt.Printf("  Images found: %d\n", imageCount)
	fmt.Printf("  Number of clusters: %d\n", actualClusters)
	fmt.Printf("  Parallel workers: %d\n", workers)
	fmt.Printf("  Python executable: %s\n", pythonPath)

	if !noPCA {
		fmt.Printf("  PCA enabled: variance ratio %.2f\n", varRatio)
	} else {
		fmt.Printf("  PCA disabled\n")
	}

	if noVisualize {
		fmt.Printf("  Visualization disabled\n")
	} else {
		fmt.Printf("  Visualization enabled\n")
	}

	fmt.Println()
}

func runClusteringPipeline() error {
	fmt.Println("Starting image clustering pipeline...")

	if err := runPythonClustering(); err != nil {
		return fmt.Errorf("clustering pipeline failed: %w", err)
	}

	fmt.Printf("\nClustering completed successfully!\n")
	fmt.Printf("Results saved to: %s\n", outputDir)

	// Show cluster summary
	if err := showClusterSummary(); err != nil {
		fmt.Printf("Warning: Could not display cluster summary: %v\n", err)
	}

	return nil
}

func runPythonClustering() any {
	panic("unimplemented")
}

func showClusterSummary() error {
	clusterDirs, err := filepath.Glob(filepath.Join(outputDir, "cluster_*"))
	if err != nil {
		return err
	}

	if len(clusterDirs) > 0 {
		fmt.Println("\nCluster Summary:")
		for _, dir := range clusterDirs {
			files, err := filepath.Glob(filepath.Join(dir, "*"))
			if err != nil {
				continue
			}
			clusterName := filepath.Base(dir)
			fmt.Printf("  %s: %d images\n", clusterName, len(files))
		}
	}

	// Check for visualization file
	vizPath := filepath.Join(outputDir, "cluster_visualization.png")
	if _, err := os.Stat(vizPath); err == nil {
		fmt.Printf("\nVisualization saved: %s\n", vizPath)
	}

	return nil
}
