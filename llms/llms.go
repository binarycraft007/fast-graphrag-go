package llms

import (
	"bytes"
	"context"
	"reflect"
	"text/template"

	"github.com/binarycraft007/fast-graphrag-go/prompts"
)

// Define a generic LLM service interface
type LLMService interface {
	SendMessage(ctx context.Context, prompt string, options ...MessageOptions) (any, error)
	GetEmbedding(ctx context.Context, texts []string, options ...MessageOptions) ([]Embedding, error)
}

func FormatAndSendPrompt(
	ctx context.Context,
	promptKey string,
	llm LLMService,
	formatArg any,
	options ...MessageOptions,
) (any, error) {
	buf := new(bytes.Buffer) // Create a new buffer
	prompt := prompts.Prompts[promptKey]
	tmpl, err := template.New(promptKey).Parse(prompt)
	if err != nil {
		return nil, err
	}
	if err = tmpl.Execute(buf, formatArg); err != nil {
		return nil, err
	}
	formatedPrompt := buf.String()
	return llm.SendMessage(ctx, formatedPrompt, options...)
}

// MessageOptions for customizing LLM requests.
type MessageConfig struct {
	Model           string
	SystemPrompt    string
	HistoryMessages []Message
	MaxTokens       int
	ResponseType    reflect.Type
	EmbeddingDim    int
}

type MessageOptions func(mc *MessageConfig)

func WithModel(model string) MessageOptions {
	return func(mc *MessageConfig) {
		mc.Model = model
	}
}

func WithSystemPrompt(systemPrompt string) MessageOptions {
	return func(mc *MessageConfig) {
		mc.SystemPrompt = systemPrompt
	}
}

func WithHistoryMessges(historyMessages []Message) MessageOptions {
	return func(mc *MessageConfig) {
		mc.HistoryMessages = historyMessages
	}
}

func WithMaxTokens(maxTokens int) MessageOptions {
	return func(mc *MessageConfig) {
		mc.MaxTokens = maxTokens
	}
}

func WithEmbeddingDim(embeddingDim int) MessageOptions {
	return func(mc *MessageConfig) {
		mc.EmbeddingDim = embeddingDim
	}
}

func WithResponseType(responseType reflect.Type) MessageOptions {
	return func(mc *MessageConfig) {
		mc.ResponseType = responseType
	}
}

type Message interface{}

// EmbeddingOptions defines parameters for generating embeddings.
type EmbeddingOptions struct {
	Model string
}

// Embedding represents a single embedding vector.
type Embedding struct {
	Vector []float32
}

// Helper function to chunk texts into smaller groups.
func chunkTexts(texts []string, chunkSize int) [][]string {
	var chunks [][]string
	for i := 0; i < len(texts); i += chunkSize {
		end := i + chunkSize
		if end > len(texts) {
			end = len(texts)
		}
		chunks = append(chunks, texts[i:end])
	}
	return chunks
}
