import '@/styles/style.css';

//Post summary, activity preview, club post, user post
//New components to make - chart, calender/log, stat container?, club page with club card component to feed preview

function ActivityIcon {
    const t = "Yup";
}

//hard to do with no auth flow
export function CardHeader({id, location, createdAt, deviceName}: {id: string; location: string; createdAt: string; deviceName?: string}) {

    return(
        <div className="card-header">
        
        </div>
    )
}

export function ActivityPreview() {
    return (
        <div className="">

        </div>
    )
}