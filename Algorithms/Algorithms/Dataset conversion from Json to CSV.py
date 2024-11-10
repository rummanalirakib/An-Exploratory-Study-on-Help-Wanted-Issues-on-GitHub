import json
import csv
import os

# Specify the folder containing the JSON files
folder_path = r'D:\Final Help Wanted Research Datset and and Code\Dataset'

# List of headers for the CSV file
headers = [
    'url', 'repository_url', 'labels_url', 'comments_url', 'events_url', 'html_url',
    'id', 'node_id', 'number', 'title', 'state', 'locked', 
    'user.login', 'user.id', 'user.node_id', 'user.avatar_url', 'user.gravatar_id',
    'user.url', 'user.html_url', 'user.followers_url', 'user.following_url',
    'user.gists_url', 'user.starred_url', 'user.subscriptions_url',
    'user.organizations_url', 'user.repos_url', 'user.events_url', 'user.received_events_url',
    'user.type', 'user.site_admin',
    'assignee.login', 'assignee.id', 'assignee.node_id', 'assignee.avatar_url',
    'assignee.gravatar_id', 'assignee.url', 'assignee.html_url', 'assignee.followers_url',
    'assignee.following_url', 'assignee.gists_url', 'assignee.starred_url',
    'assignee.subscriptions_url', 'assignee.organizations_url', 'assignee.repos_url',
    'assignee.events_url', 'assignee.received_events_url', 'assignee.type',
    'assignee.site_admin',
    'assignees.login', 'assignees.id', 'assignees.node_id', 'assignees.avatar_url',
    'assignees.gravatar_id', 'assignees.url', 'assignees.html_url',
    'assignees.followers_url', 'assignees.following_url', 'assignees.gists_url',
    'assignees.starred_url', 'assignees.subscriptions_url', 'assignees.organizations_url',
    'assignees.repos_url', 'assignees.events_url', 'assignees.received_events_url',
    'assignees.type', 'assignees.site_admin',
    'milestone.url', 'milestone.html_url', 'milestone.labels_url', 'milestone.id',
    'milestone.node_id', 'milestone.number', 'milestone.title', 'milestone.description',
    'milestone.creator', 'milestone.open_issues', 'milestone.closed_issues',
    'milestone.state', 'milestone.created_at', 'milestone.updated_at', 'milestone.due_on',
    'milestone.closed_at',
    'comments_count', 'created_at', 'updated_at', 'closed_at',
    'author_association', 'active_lock_reason', 'body',
    'reactions.url', 'reactions.total_count', 'reactions.+1', 'reactions.-1',
    'reactions.laugh', 'reactions.hooray', 'reactions.confused', 'reactions.heart',
    'reactions.rocket', 'reactions.eyes', 'timeline_url',
    'performed_via_github_app.id', 'performed_via_github_app.slug', 'performed_via_github_app.node_id',
    'performed_via_github_app.owner', 'performed_via_github_app.name', 'performed_via_github_app.description',
    'performed_via_github_app.external_url', 'performed_via_github_app.html_url',
    'performed_via_github_app.created_at', 'performed_via_github_app.updated_at',
    'performed_via_github_app.permissions', 'performed_via_github_app.events',
    'state_reason', 'milestone', 'draft',
    'pull_request.url', 'pull_request.html_url', 'pull_request.diff_url',
    'pull_request.patch_url', 'pull_request.merged_at',
    'labels.id', 'labels.node_id', 'labels.url', 'labels.name',
    'labels.color', 'labels.default', 'labels.description'
]



for folder_name in os.listdir(folder_path):
    print(folder_name)
    csv_files = folder_name + ".csv"
    destination_file = os.path.join('D:\Final Help Wanted Research Datset and and Code\HW Issues in csv file', csv_files)
    # Create and write to the CSV file
    with open(destination_file, mode='w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=headers, extrasaction='ignore', escapechar='\\')
        writer.writeheader()
        
        issue_folder_path = os.path.join(folder_path, folder_name)
        issue_folder_path = os.path.join(issue_folder_path, 'issues')
        print(issue_folder_path)
        # Process each JSON file in the folder
        for json_file in os.listdir(issue_folder_path):
            if json_file.endswith('.json'):
                with open(os.path.join(issue_folder_path, json_file), 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    
                    for item in data:
                        row = {}
                        
                        # Simple fields directly from item
                        row['url'] = item.get('url')
                        row['repository_url'] = item.get('repository_url')
                        row['labels_url'] = item.get('labels_url')
                        row['comments_url'] = item.get('comments_url')
                        row['events_url'] = item.get('events_url')
                        row['html_url'] = item.get('html_url')
                        row['id'] = item.get('id')
                        row['node_id'] = item.get('node_id')
                        row['number'] = item.get('number')
                        row['title'] = item.get('title')
                        row['state'] = item.get('state')
                        row['locked'] = item.get('locked')
                        row['comments_count'] = item.get('comments')
                        row['created_at'] = item.get('created_at')
                        row['updated_at'] = item.get('updated_at')
                        row['closed_at'] = item.get('closed_at')
                        row['author_association'] = item.get('author_association')
                        row['active_lock_reason'] = item.get('active_lock_reason')
                        row['body'] = item.get('body')
                        row['state_reason'] = item.get('state_reason')
                        row['milestone'] = item.get('milestone')
                        row['draft'] = item.get('draft')
                        row['timeline_url'] = item.get('timeline_url')
                        
                        # User fields
                        user = item.get('user', {})
                        row['user.login'] = user.get('login')
                        row['user.id'] = user.get('id')
                        row['user.node_id'] = user.get('node_id')
                        row['user.avatar_url'] = user.get('avatar_url')
                        row['user.gravatar_id'] = user.get('gravatar_id')
                        row['user.url'] = user.get('url')
                        row['user.html_url'] = user.get('html_url')
                        row['user.followers_url'] = user.get('followers_url')
                        row['user.following_url'] = user.get('following_url')
                        row['user.gists_url'] = user.get('gists_url')
                        row['user.starred_url'] = user.get('starred_url')
                        row['user.subscriptions_url'] = user.get('subscriptions_url')
                        row['user.organizations_url'] = user.get('organizations_url')
                        row['user.repos_url'] = user.get('repos_url')
                        row['user.events_url'] = user.get('events_url')
                        row['user.received_events_url'] = user.get('received_events_url')
                        row['user.type'] = user.get('type')
                        row['user.site_admin'] = user.get('site_admin')
                        
                        # Assignee fields
                        assignee = item.get('assignee')
                        if assignee:
                            row['assignee.login'] = assignee.get('login')
                            row['assignee.id'] = assignee.get('id')
                            row['assignee.node_id'] = assignee.get('node_id')
                            row['assignee.avatar_url'] = assignee.get('avatar_url')
                            row['assignee.gravatar_id'] = assignee.get('gravatar_id')
                            row['assignee.url'] = assignee.get('url')
                            row['assignee.html_url'] = assignee.get('html_url')
                            row['assignee.followers_url'] = assignee.get('followers_url')
                            row['assignee.following_url'] = assignee.get('following_url')
                            row['assignee.gists_url'] = assignee.get('gists_url')
                            row['assignee.starred_url'] = assignee.get('starred_url')
                            row['assignee.subscriptions_url'] = assignee.get('subscriptions_url')
                            row['assignee.organizations_url'] = assignee.get('organizations_url')
                            row['assignee.repos_url'] = assignee.get('repos_url')
                            row['assignee.events_url'] = assignee.get('events_url')
                            row['assignee.received_events_url'] = assignee.get('received_events_url')
                            row['assignee.type'] = assignee.get('type')
                            row['assignee.site_admin'] = assignee.get('site_admin')
                        else:
                            row['assignee.login'] = None
                            row['assignee.id'] = None
                            row['assignee.node_id'] = None
                            row['assignee.avatar_url'] = None
                            row['assignee.gravatar_id'] = None
                            row['assignee.url'] = None
                            row['assignee.html_url'] = None
                            row['assignee.followers_url'] = None
                            row['assignee.following_url'] = None
                            row['assignee.gists_url'] = None
                            row['assignee.starred_url'] = None
                            row['assignee.subscriptions_url'] = None
                            row['assignee.organizations_url'] = None
                            row['assignee.repos_url'] = None
                            row['assignee.events_url'] = None
                            row['assignee.received_events_url'] = None
                            row['assignee.type'] = None
                            row['assignee.site_admin'] = None
                        
                        # Assignees fields (list of dicts)
                        assignees = item.get('assignees', [])
                        row['assignees.login'] = [a.get('login') for a in assignees]
                        row['assignees.id'] = [a.get('id') for a in assignees]
                        row['assignees.node_id'] = [a.get('node_id') for a in assignees]
                        row['assignees.avatar_url'] = [a.get('avatar_url') for a in assignees]
                        row['assignees.gravatar_id'] = [a.get('gravatar_id') for a in assignees]
                        row['assignees.url'] = [a.get('url') for a in assignees]
                        row['assignees.html_url'] = [a.get('html_url') for a in assignees]
                        row['assignees.followers_url'] = [a.get('followers_url') for a in assignees]
                        row['assignees.following_url'] = [a.get('following_url') for a in assignees]
                        row['assignees.gists_url'] = [a.get('gists_url') for a in assignees]
                        row['assignees.starred_url'] = [a.get('starred_url') for a in assignees]
                        row['assignees.subscriptions_url'] = [a.get('subscriptions_url') for a in assignees]
                        row['assignees.organizations_url'] = [a.get('organizations_url') for a in assignees]
                        row['assignees.repos_url'] = [a.get('repos_url') for a in assignees]
                        row['assignees.events_url'] = [a.get('events_url') for a in assignees]
                        row['assignees.received_events_url'] = [a.get('received_events_url') for a in assignees]
                        row['assignees.type'] = [a.get('type') for a in assignees]
                        row['assignees.site_admin'] = [a.get('site_admin') for a in assignees]
                        
                        if len(row['assignees.login']) == 0:
                            row['assignees.login'] = None
                        
                        if len(row['assignees.id']) == 0:
                            row['assignees.id'] = None
                        
                        if len(row['assignees.node_id']) == 0:
                            row['assignees.node_id'] = None
                        
                        if len(row['assignees.avatar_url']) == 0:
                            row['assignees.avatar_url'] = None
                        
                        if len(row['assignees.gravatar_id']) == 0:
                            row['assignees.gravatar_id'] = None
                        
                        if len(row['assignees.url']) == 0:
                            row['assignees.url'] = None
                        
                        if len(row['assignees.html_url']) == 0:
                            row['assignees.html_url'] = None
                        
                        if len(row['assignees.followers_url']) == 0:
                            row['assignees.followers_url'] = None
                        
                        if len(row['assignees.following_url']) == 0:
                            row['assignees.following_url'] = None
                        
                        if len(row['assignees.gists_url']) == 0:
                            row['assignees.gists_url'] = None
                        
                        if len(row['assignees.starred_url']) == 0:
                            row['assignees.starred_url'] = None
                        
                        if len(row['assignees.subscriptions_url']) == 0:
                            row['assignees.subscriptions_url'] = None
                        
                        if len(row['assignees.organizations_url']) == 0:
                            row['assignees.organizations_url'] = None
                        
                        if len(row['assignees.repos_url']) == 0:
                            row['assignees.repos_url'] = None
                        
                        if len(row['assignees.events_url']) == 0:
                            row['assignees.events_url'] = None
                        
                        if len(row['assignees.received_events_url']) == 0:
                            row['assignees.received_events_url'] = None
                        
                        if len(row['assignees.type']) == 0:
                            row['assignees.type'] = None
                        
                        if len(row['assignees.site_admin']) == 0:
                            row['assignees.site_admin'] = None
                        # Milestone fields
                        milestone = item.get('milestone')
                        if milestone:
                            row['milestone.url'] = milestone.get('url')
                            row['milestone.html_url'] = milestone.get('html_url')
                            row['milestone.labels_url'] = milestone.get('labels_url')
                            row['milestone.id'] = milestone.get('id')
                            row['milestone.node_id'] = milestone.get('node_id')
                            row['milestone.number'] = milestone.get('number')
                            row['milestone.title'] = milestone.get('title')
                            row['milestone.description'] = milestone.get('description')
                            row['milestone.creator'] = milestone.get('creator')
                            row['milestone.open_issues'] = milestone.get('open_issues')
                            row['milestone.closed_issues'] = milestone.get('closed_issues')
                            row['milestone.state'] = milestone.get('state')
                            row['milestone.created_at'] = milestone.get('created_at')
                            row['milestone.updated_at'] = milestone.get('updated_at')
                            row['milestone.due_on'] = milestone.get('due_on')
                            row['milestone.closed_at'] = milestone.get('closed_at')
                        else:
                            row['milestone.url'] = None
                            row['milestone.html_url'] = None
                            row['milestone.labels_url'] = None
                            row['milestone.id'] = None
                            row['milestone.node_id'] = None
                            row['milestone.number'] = None
                            row['milestone.title'] = None
                            row['milestone.description'] = None
                            row['milestone.creator'] = None
                            row['milestone.open_issues'] = None
                            row['milestone.closed_issues'] = None
                            row['milestone.state'] = None
                            row['milestone.created_at'] = None
                            row['milestone.updated_at'] = None
                            row['milestone.due_on'] = None
                            row['milestone.closed_at'] = None
                        
                        # Reactions fields
                        reactions = item.get('reactions', {})
                        row['reactions.url'] = reactions.get('url')
                        row['reactions.total_count'] = reactions.get('total_count')
                        row['reactions.+1'] = reactions.get('+1')
                        row['reactions.-1'] = reactions.get('-1')
                        row['reactions.laugh'] = reactions.get('laugh')
                        row['reactions.hooray'] = reactions.get('hooray')
                        row['reactions.confused'] = reactions.get('confused')
                        row['reactions.heart'] = reactions.get('heart')
                        row['reactions.rocket'] = reactions.get('rocket')
                        row['reactions.eyes'] = reactions.get('eyes')
                        
                        # Performed via GitHub app fields
                        performed_via_github_app = item.get('performed_via_github_app')
                        if performed_via_github_app:
                            row['performed_via_github_app.id'] = performed_via_github_app.get('id')
                            row['performed_via_github_app.slug'] = performed_via_github_app.get('slug')
                            row['performed_via_github_app.node_id'] = performed_via_github_app.get('node_id')
                            row['performed_via_github_app.owner'] = performed_via_github_app.get('owner')
                            row['performed_via_github_app.name'] = performed_via_github_app.get('name')
                            row['performed_via_github_app.description'] = performed_via_github_app.get('description')
                            row['performed_via_github_app.external_url'] = performed_via_github_app.get('external_url')
                            row['performed_via_github_app.html_url'] = performed_via_github_app.get('html_url')
                            row['performed_via_github_app.created_at'] = performed_via_github_app.get('created_at')
                            row['performed_via_github_app.updated_at'] = performed_via_github_app.get('updated_at')
                            row['performed_via_github_app.permissions'] = performed_via_github_app.get('permissions')
                            row['performed_via_github_app.events'] = performed_via_github_app.get('events')
                        else:
                            row['performed_via_github_app.id'] = None
                            row['performed_via_github_app.slug'] = None
                            row['performed_via_github_app.node_id'] = None
                            row['performed_via_github_app.owner'] = None
                            row['performed_via_github_app.name'] = None
                            row['performed_via_github_app.description'] = None
                            row['performed_via_github_app.external_url'] = None
                            row['performed_via_github_app.html_url'] = None
                            row['performed_via_github_app.created_at'] = None
                            row['performed_via_github_app.updated_at'] = None
                            row['performed_via_github_app.permissions'] = None
                            row['performed_via_github_app.events'] = None
                                            
                        # Pull request fields
                        pull_request = item.get('pull_request', {})
                        row['pull_request.url'] = pull_request.get('url')
                        row['pull_request.html_url'] = pull_request.get('html_url')
                        row['pull_request.diff_url'] = pull_request.get('diff_url')
                        row['pull_request.patch_url'] = pull_request.get('patch_url')
                        row['pull_request.merged_at'] = pull_request.get('merged_at')
                        
                        # Labels fields
                        labels = item.get('labels', [])
                        row['labels.id'] = [label.get('id') for label in labels]
                        row['labels.node_id'] = [label.get('node_id') for label in labels]
                        row['labels.url'] = [label.get('url') for label in labels]
                        labels = item.get('labels', [])
                        labels_name = [f'<{label.get("name" , "")}>' for label in labels]
                        row['labels.name'] = ''.join(labels_name)
                        row['labels.color'] = [label.get('color') for label in labels]
                        row['labels.default'] = [label.get('default') for label in labels]
                        row['labels.description'] = [label.get('description') for label in labels]
                        
                        if len(row['labels.id']) == 0:
                            row['labels.id'] = None
                        
                        if len(row['labels.node_id']) == 0:
                            row['labels.node_id'] = None
                        
                        if len(row['labels.url']) == 0:
                            row['labels.url'] = None
                        
                        if len(row['labels.name']) == 0:
                            row['labels.name'] = None
                        
                        if len(row['labels.color']) == 0:
                            row['labels.color'] = None
                        
                        if len(row['labels.default']) == 0:
                            row['labels.default'] = None
                        
                        if len(row['labels.description']) == 0:
                            row['labels.description'] = None
                        
                        # Write the row to CSV
                        writer.writerow(row)

print("Data has been written to output.csv")
