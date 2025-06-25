import type { QuerySchema, ResponseSchema } from './types/schema';

export async function askQuery(payload: QuerySchema): Promise<ResponseSchema> {
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
