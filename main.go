package main

import (
	"fmt"
	"os"
	"runtime"

	"github.com/spf13/cobra"
)

var version = "1.0.0"

func main() {
	if err := rootCmd.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

var rootCmd = &cobra.Command{
	Use:   "imgcli",
	Short: "Image clustering CLI tool",
	Long: `imgcli is a command-line tool for clustering images based on visual similarity.
It uses ResNet50 for feature extraction and K-means clustering to group similar images.

The tool processes images in parallel using Go for I/O operations and Python for 
machine learning computations. Results are organized into cluster-specific folders.`,
	Version: version,
}

var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "Print the version information",
	Run: func(cmd *cobra.Command, args []string) {
		fmt.Printf("imgcli version %s\n", version)
		fmt.Printf("Go version: %s\n", runtime.Version())
		fmt.Printf("OS/Arch: %s/%s\n", runtime.GOOS, runtime.GOARCH)
	},
}

func init() {
	rootCmd.AddCommand(clusterCmd)
	rootCmd.AddCommand(versionCmd)

	// Global flags
	rootCmd.PersistentFlags().BoolP("verbose", "v", false, "Enable verbose output")

	// Set version template
	rootCmd.SetVersionTemplate(`{{printf "%s version %s" .Name .Version}}
`)
}
