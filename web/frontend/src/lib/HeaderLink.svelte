<script lang="ts">
	import { page } from '$app/stores';
	export let href: string;
	export let text: string;

	let inactiveClasses = 'bg-gray-100 text-blue-600 hover:bg-blue-700 hover:text-white';
	let activeClasses = 'bg-blue-600 text-white hover:bg-blue-700';
</script>

<a
	{href}
	class={`px-4 py-2 rounded-md font-medium transition-colors duration-200 ${
		$page.url.pathname === href ? activeClasses : inactiveClasses
	}`}
>
	{text}
</a>
