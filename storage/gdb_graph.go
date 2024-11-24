package storage

// BaseGraphStorage defines the interface for graph storage.
type BaseGraphStorage[Node, ID any] interface {
	InsertStart() error
	InsertDone() error
}
