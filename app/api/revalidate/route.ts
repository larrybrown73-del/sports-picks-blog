import { NextRequest, NextResponse } from "next/server";
import { revalidatePath } from "next/cache";

export async function POST(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const token = searchParams.get("secret");

    if (token !== process.env.REVALIDATION_SECRET) {
      return NextResponse.json(
        { message: "Invalid token unauthorized" },
        { status: 401 }
      );
    }

    const path = searchParams.get("path") || "/";
    revalidatePath(path);

    return NextResponse.json({ revalidated: true, now: Date.now() });
  } catch (err) {
    return NextResponse.json(
      { message: "Error revalidating", error: err },
      { status: 500 }
    );
  }
}
