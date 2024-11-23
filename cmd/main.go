package main

import (
	"context"
	"os"
	"strings"
	"time"

	_ "embed"

	"github.com/binarycraft007/fast-graphrag-go/llms"
	"github.com/binarycraft007/fast-graphrag-go/services"
	"github.com/binarycraft007/fast-graphrag-go/types"
)

//go:embed book.txt
var data string

func main() {
	entiryTypes := []string{"Character", "Animal", "Place", "Object", "Activity", "Event"}
	ctx := context.Background()
	llm, err := llms.NewGoogleAILLMService(ctx, os.Getenv("GEMINI_API_KEY"))
	if err != nil {
		panic(err)
	}
	defer llm.Client.Close()
	chunkService := services.NewDefaultChunkingService()
	chunks := chunkService.Extract([]types.Document{
		{Data: data},
	})
	infoExtract := services.DefaultInformationExtractionService{}
	_, err = infoExtract.Extract(
		llm,
		chunks,
		map[string]any{
			"domain": "Analyze this story and identify the characters. Focus on how they interact with each other, the locations they explore, and their relationships.",
			"example_queries": []string{
				"What is the significance of Christmas Eve in A Christmas Carol?",
				"How does the setting of Victorian London contribute to the story's themes?",
				"Describe the chain of events that leads to Scrooge's transformation.",
				"How does Dickens use the different spirits (Past, Present, and Future) to guide Scrooge?",
				"Why does Dickens choose to divide the story into \"staves\" rather than chapters?",
			},
			"entity_types": strings.Join(entiryTypes, ","),
		},
		entiryTypes,
	)
	if err != nil {
		panic(err)
	}
	for {
		time.Sleep(time.Second * 1)
	}
}
