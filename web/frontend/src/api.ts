import type { MessageResponseSchema, QuerySchema } from './types/schema';

export async function askQuery(payload: QuerySchema): Promise<MessageResponseSchema> {
	const response = await fetch('/api/v1/ask/query', {
		method: 'POST',
		credentials: 'same-origin',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({ query: payload.query })
	});

	if (!response.ok) {
		throw new Error(`API error: ${response.status} ${response.statusText}`);
	}

	return response.json();
}

export async function loadHistory(): Promise<string[]> {
	const response = await fetch('/api/v1/ask/history', {
		method: 'GET',
		credentials: 'same-origin'
	});

	if (!response.ok) {
		throw new Error(`API error: ${response.status} ${response.statusText}`);
	}

	return response.json();
}

export async function deleteHistory(): Promise<void> {
	const response = await fetch('/api/v1/ask/history', {
		method: 'DELETE',
		credentials: 'same-origin'
	});

	if (!response.ok) {
		throw new Error(`API error: ${response.status} ${response.statusText}`);
	}
}
