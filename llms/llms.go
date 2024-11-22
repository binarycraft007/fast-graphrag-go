package llms

import "context"

type LLM interface {
	SendMessage(
		ctx context.Context,
		prompt string,
		systemPrompt string,
		historyMessages interface{},
		response any,
	) (interface{}, error)
}

type Embedding interface{}
